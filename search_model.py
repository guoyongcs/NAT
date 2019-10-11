import genotypes
from operations import *
import utils
import numpy as np
from utils import arch_to_genotype, draw_genotype
import os
from pygcn.layers import GraphConvolution

class NASOp(nn.Module):
    def __init__(self, C, stride, op_type):
        super(NASOp, self).__init__()
        self._ops = nn.ModuleList()
        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % op_type)
        except:
            assert False, 'not supported op type %s' % (op_type)
        for primitive in COMPACT_PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)


class NASCell(nn.Module):
    def __init__(self, steps, device, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, loose_end=False, concat=None, op_type='FULLY_CONCAT_PRIMITIVES'):
        super(NASCell, self).__init__()
        self.steps = steps
        self.device = device
        self.multiplier = multiplier
        self.C = C
        self.reduction = reduction
        self.loose_end = loose_end
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                op = NASOp(C, stride, op_type)
                self._ops.append(op)

        self._concat = concat

    def forward(self, s0, s1, arch):
        """

        :param s0:
        :param s1:
        :param arch: a list, the element is (op_id, from_node, to_node), sorted by to_node (!!not check
                     the ordering for efficiency, but must be assured when generating!!)
                     from_node/to_node starts from 0, 0 is the prev_prev_node, 1 is prev_node
                     The mapping from (F, T) pair to edge_ID is (T-2)(T+1)/2+S,

        :return:
        """
        s0 = self.preprocess0.forward(s0)
        s1 = self.preprocess1.forward(s1)
        states = {0: s0, 1: s1}
        used_nodes = set()
        for op, f, t in arch:
            edge_id = int((t - 2) * (t + 1) / 2 + f)
            if t in states:
                states[t] = states[t] + self._ops[edge_id](states[f], op)
            else:
                states[t] = self._ops[edge_id](states[f], op)
            used_nodes.add(f)
        if self._concat is not None:
            state_list = []
            for i in range(2, self._steps + 2):
                if i in self._concat:
                    # in concat list
                    state_list.append(states[i])
                else:
                    # not in concat list, we multiply 0
                    state_list.append(states[i] * 0)
            return torch.cat(state_list, dim=1)
        else:
            if self.loose_end:
                state_list = []
                for i in range(2, self._steps + 2):
                    if i not in used_nodes:
                        # loose end
                        state_list.append(states[i])
                    else:
                        # not loose end
                        state_list.append(states[i] * 0)
                return torch.cat(state_list, dim=1)
            else:
                return torch.cat([states[i] for i in range(2, self._steps + 2)], dim=1)


class ArchTransformer(nn.Module):
    def __init__(self, steps, device, edge_hid, nfeat, gcn_hid, dropout, normalize=False,
                 op_type='FULLY_CONCAT_PRIMITIVES'):
        """

        :param nfeat: feature dimension of each node in the graph
        :param nhid: hidden dimension
        :param dropout: dropout rate for GCN
        """
        super(ArchTransformer, self).__init__()
        self.steps = steps
        self.device = device
        self.normalize = normalize
        self.op_type = op_type
        num_ops = len(genotypes.TRANSFORM_PRIMITIVES)
        self.gc1 = GraphConvolution(nfeat, gcn_hid)
        self.gc2 = GraphConvolution(gcn_hid, gcn_hid)
        self.dropout = dropout
        self.fc = nn.Linear(gcn_hid, num_ops * 2)

        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % op_type)
        except:
            assert False, 'not supported op type %s' % (op_type)

        # the first two nodes
        self.node_hidden = nn.Embedding(2, 2*edge_hid)
        self.op_hidden = nn.Embedding(len(COMPACT_PRIMITIVES), edge_hid)
        # [op0, op1]
        self.emb_attn = nn.Linear(2*edge_hid, nfeat)

    def forward(self, arch):
        # initial the first two nodes
        op0_list = []
        op1_list = []
        for idx, (op, f, t) in enumerate(arch):
            if idx%2 == 0:
                op0_list.append(op)
            else:
                op1_list.append(op)
        assert len(op0_list) == len(op1_list), 'inconsistent size between op0_list and op1_list'
        node_list = utils.get_variable(list(range(0, 2, 1)), self.device, requires_grad=False)
        op0_list = utils.get_variable(op0_list, self.device, requires_grad=False)
        op1_list = utils.get_variable(op1_list, self.device, requires_grad=False)
        # first two nodes
        x_node_hidden = self.node_hidden(node_list)
        x_op0_hidden = self.op_hidden(op0_list)
        x_op1_hidden = self.op_hidden(op1_list)
        '''
            node0
            node1
            op0, op1
        '''
        x_op_hidden = torch.cat((x_op0_hidden, x_op1_hidden), dim=1)
        x_hidden = torch.cat((x_node_hidden, x_op_hidden), dim=0)
        # initialize x and adj
        x = self.emb_attn(x_hidden)
        adj = utils.parse_arch(arch, self.steps+2).to(self.device)
        # normalize features and adj
        if self.normalize:
            x = utils.sum_normalize(x)
            adj = utils.sum_normalize(adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = x[2:]
        logits = self.fc(x)
        logits = logits.view(self.steps*2, -1)
        probs = F.softmax(logits, dim=-1)
        probs = probs + 1e-5
        log_probs = torch.log(probs)
        action = probs.multinomial(num_samples=1)
        selected_log_p = log_probs.gather(-1, action)
        log_p = selected_log_p.sum()
        entropy = -(log_probs * probs).sum()
        arch = utils.translate_arch(arch, action, self.op_type)
        return arch, log_p, entropy


class ArchMaster(nn.Module):
    def __init__(self, n_ops, n_nodes, device, controller_hid=None, lstm_num_layers=2):
        super(ArchMaster, self).__init__()
        self.K = sum([x + 2 for x in range(n_nodes)])
        self.n_ops = n_ops
        self.n_nodes = n_nodes
        self.device = device

        self.controller_hid = controller_hid
        self.attention_hid = self.controller_hid
        self.lstm_num_layers = lstm_num_layers

        # Embedding of (n_nodes+1) nodes
        # Note that the (n_nodes+2)-th node will not be used
        self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
        self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
        self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
        self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
        self.tanh = nn.Tanh()
        self.prev_nodes, self.prev_ops = [], []
        self.query_index = torch.LongTensor(range(0, n_nodes+1)).to(device)

    def _get_default_hidden(self, key):
        return utils.get_variable(
            torch.zeros(key, self.controller_hid), self.device, requires_grad=False)

    # device
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.device, requires_grad=False),
                utils.get_variable(zeros.clone(), self.device, requires_grad=False))

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.w_soft.bias.data.fill_(0)

    def forward(self):
        self.prev_nodes, self.prev_ops = [], []
        batch_size = 1
        inputs = self.static_inputs[batch_size]  # batch_size x hidden_dim
        for node_idx in range(self.n_nodes):
            for i in range(2):  # index_1, index_2
                if node_idx == 0 and i == 0:
                    embed = inputs
                else:
                    embed = self.node_op_hidden(inputs)
                # force uniform
                probs = F.softmax(torch.zeros(node_idx + 2).type_as(embed), dim=-1)
                action = probs.multinomial(num_samples=1)
                self.prev_nodes.append(action)
                inputs = utils.get_variable(action, self.device, requires_grad=False)
            for i in range(2):  # op_1, op_2
                embed = self.node_op_hidden(inputs)
                # force uniform
                probs = F.softmax(torch.zeros(self.n_ops).type_as(embed), dim=-1)
                action = probs.multinomial(num_samples=1)
                self.prev_ops.append(action)
                inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
        arch = utils.convert_lstm_output(self.n_nodes, torch.cat(self.prev_nodes), torch.cat(self.prev_ops))
        return arch


class NASNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, device, steps=4, stem_multiplier=3, controller_hid=None, entropy_coeff=[0.0, 0.0], edge_hid=100, transformer_nfeat=1024, transformer_nhid=512, transformer_dropout=0, transformer_normalize=False, loose_end=False, normal_concat=None, reduce_concat=None, op_type='FULLY_CONCAT_PRIMITIVES'):
        super(NASNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        multiplier = steps
        self._device = device

        self.controller_hid = controller_hid
        self.entropy_coeff = entropy_coeff

        self.edge_hid = edge_hid
        self.transformer_nfeat = transformer_nfeat
        self.transformer_nhid = transformer_nhid
        self.transformer_dropout = transformer_dropout
        self.transformer_normalize = transformer_normalize
        self.op_type = op_type

        self.loose_end = loose_end

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        _concat = None
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                if reduce_concat is not None:
                    _concat = reduce_concat
            else:
                reduction = False
                if normal_concat is not None:
                    _concat = normal_concat
            cell = NASCell(steps, device, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, loose_end=loose_end, concat=_concat, op_type=self.op_type)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_arch_master()
        self._initialize_arch_transformer()
        self.flag = 0

    def _initialize_arch_master(self):
        try:
            COMPACT_PRIMITIVES = eval("genotypes.%s" % self.op_type)
        except:
            assert False, 'not supported op type %s' % (self.op_type)

        num_ops = len(COMPACT_PRIMITIVES)-1
        self.arch_normal_master = ArchMaster(num_ops, self._steps, self._device, self.controller_hid)
        self.arch_reduce_master = ArchMaster(num_ops, self._steps, self._device, self.controller_hid)
        self._arch_parameters = list(self.arch_normal_master.parameters()) + list(self.arch_reduce_master.parameters())

    def _initialize_arch_transformer(self):
        self.arch_transformer = ArchTransformer(self._steps, self._device, self.edge_hid, self.transformer_nfeat, self.transformer_nhid, self.transformer_dropout, self.transformer_normalize, op_type=self.op_type)
        self._transformer_parameters = list(self.arch_transformer.parameters())

    def _inner_forward(self, input, arch_normal, arch_reduce):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                archs = arch_reduce
            else:
                archs = arch_normal
            s0, s1 = s1, cell(s0, s1, archs)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _test_acc(self, test_queue, arch_normal, arch_reduce):
        # go over all the testing data to obtain the accuracy
        top1 = utils.AvgrageMeter()
        for step, (test_input, test_target) in enumerate(test_queue):
            test_input = test_input.to(self._device)
            test_target = test_target.to(self._device)
            n = test_input.size(0)
            logits = self._inner_forward(test_input, arch_normal, arch_reduce)
            accuracy = utils.accuracy(logits, test_target)[0]
            top1.update(accuracy.item(), n)
        return top1.avg

    def test(self, test_queue, n_optim, logger, folder, suffix):
        arch_normal = self.arch_normal_master()
        arch_reduce = self.arch_reduce_master()
        self.derive_optimized_arch(test_queue, arch_normal, arch_reduce, n_optim, logger, folder, suffix)

    def derive_optimized_arch(self, test_queue, arch_normal, arch_reduce, n_optim, logger, folder, suffix, normal_concat=None, reduce_concat=None):
        best_acc = -np.inf
        best_optimized_acc = -np.inf
        best_arch_normal_logP = None
        best_arch_reduce_logP = None
        best_arch_normal_ent = None
        best_arch_reduce_ent = None
        best_arch_normal = None
        best_arch_reduce = None
        best_optimized_arch_normal = None
        best_optimized_arch_reduce = None

        top1 = self._test_acc(test_queue, arch_normal, arch_reduce)
        logger.info("Sampling candidate architectures ...")
        for i in range(n_optim):
            optimized_normal, optimized_normal_logP, optimized_normal_entropy = self.arch_transformer.forward(arch_normal)
            optimized_reduce, optimized_reduce_logP, optimized_reduce_entropy = self.arch_transformer.forward(arch_reduce)
            optimized_top1 = self._test_acc(test_queue, optimized_normal, optimized_reduce)
            if optimized_top1 > best_optimized_acc:
                best_acc = top1
                best_optimized_acc = optimized_top1
                best_arch_normal = arch_normal
                best_arch_reduce = arch_reduce
                best_optimized_arch_normal = optimized_normal
                best_optimized_arch_reduce = optimized_reduce
                best_arch_normal_logP = optimized_normal_logP
                best_arch_reduce_logP = optimized_reduce_logP
                best_arch_normal_ent = optimized_normal_entropy
                best_arch_reduce_ent = optimized_reduce_entropy

        logger.info("Best: Accuracy %f OptimizedAccuracy %f -LogP %f ENT %f", best_acc, best_optimized_acc,
                    -best_arch_normal_logP - best_arch_reduce_logP, best_arch_normal_ent + best_arch_reduce_ent)
        logger.info("Normal: \n%s\n%s", best_arch_normal, best_optimized_arch_normal)
        logger.info("Reduction: \n%s\n%s", best_arch_reduce, best_optimized_arch_reduce)
        genotype = arch_to_genotype(best_arch_normal, best_arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        transformed_genotype = arch_to_genotype(best_optimized_arch_normal, best_optimized_arch_reduce, self._steps, self.op_type, normal_concat, reduce_concat)
        draw_genotype(genotype.normal, self._steps, os.path.join(folder, "normal_%s" % suffix), genotype.normal_concat)
        draw_genotype(genotype.reduce, self._steps, os.path.join(folder, "reduce_%s" % suffix), genotype.reduce_concat)
        draw_genotype(transformed_genotype.normal, self._steps, os.path.join(folder, "optimized_normal_%s" % suffix), transformed_genotype.normal_concat)
        draw_genotype(transformed_genotype.reduce, self._steps, os.path.join(folder, "optimized_reduce_%s" % suffix), transformed_genotype.reduce_concat)
        logger.info('genotype = %s', genotype)
        logger.info('optimized_genotype = %s', transformed_genotype)

    def transformer_forward(self, valid_input):
        arch_normal = self.arch_normal_master.forward()
        arch_reduce = self.arch_reduce_master.forward()
        optimized_normal, optimized_normal_logP, optimized_normal_entropy = self.arch_transformer.forward(arch_normal)
        optimized_reduce, optimized_reduce_logP, optimized_reduce_entropy = self.arch_transformer.forward(arch_reduce)
        logits = self._inner_forward(valid_input, arch_normal, arch_reduce)
        optimized_logits = self._inner_forward(valid_input, optimized_normal, optimized_reduce)
        return logits, optimized_logits, optimized_normal, optimized_normal_logP, optimized_normal_entropy, optimized_reduce, optimized_reduce_logP, optimized_reduce_entropy

    def step(self, valid_input, valid_target):
        arch_normal = self.arch_normal_master.forward()
        arch_reduce = self.arch_reduce_master.forward()
        self._model_optimizer.zero_grad()
        logits = self._inner_forward(valid_input, arch_normal, arch_reduce)
        loss = self._criterion(logits, valid_target)
        loss.backward()
        self._model_optimizer.step()
        return logits, loss

    def _loss_transformer(self, input, target, baseline=None):
        logits, optimized_logits, optimized_normal, optimized_normal_logP, optimized_normal_entropy, optimized_reduce, optimized_reduce_logP, optimized_reduce_entropy = self.transformer_forward(input)
        accuracy = utils.accuracy(logits, target)[0] / 100.0
        optimized_accuracy = utils.accuracy(optimized_logits, target)[0] / 100.0
        reward_old = optimized_accuracy - accuracy
        reward_old = reward_old if reward_old>0 else reward_old
        reward = reward_old - baseline if baseline else reward_old
        policy_loss = -(optimized_normal_logP + optimized_reduce_logP) * reward - (
        self.entropy_coeff[0] * optimized_normal_entropy + self.entropy_coeff[1] * optimized_reduce_entropy)
        return policy_loss, reward, optimized_accuracy, optimized_normal_entropy, optimized_reduce_entropy

    def arch_parameters(self):
        return self._arch_parameters

    def transformer_parameters(self):
        return self._transformer_parameters

    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch' not in k:
                yield v

