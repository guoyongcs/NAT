import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import itertools
from genotypes import LOOSE_END_PRIMITIVES, FULLY_CONCAT_PRIMITIVES, TRANSFORM_PRIMITIVES, Genotype
from graphviz import Digraph
from collections import defaultdict
import scipy.sparse as sp
import torch.nn as nn

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def imagewise_accuracy(output, target, pid):

    res = {}
    r = zip(output, target, pid)
    for o, t, p in r:
        tokens = p.split('_')
        organ = tokens[0]
        prob = o[t]
        if organ in res:
            res[organ].append(float(prob>=0.5))
        else:
            res[organ] = [float(prob>=0.5)]

    result = {}
    all = 0
    n = 0
    for k, v in res.items():
        s = np.sum(v)
        m = np.mean(v)
        result[k] = m
        all += s
        n += len(v)
    all_mean = all / n
    result['all'] = all_mean
    return result


def subjectiwise_accuracy(output, target, pid):

    res = {}
    r = zip(output, target, pid)
    for key, value in itertools.groupby(r, key=lambda x:x[-1]):
        tokens = key.split('_')
        organ = tokens[0]
        patentID = tokens[1]
        prob = [p[l] for p, l, id in value]
        max_p = max(prob)
        min_p = min(prob)
        mean_p = sum(prob)/len(prob)
        if organ in res:
            res[organ].append([float(max_p>=0.5), float(min_p>=0.5), float(mean_p>=0.5)])
        else:
            res[organ] = [[float(max_p>=0.5), float(min_p>=0.5), float(mean_p>=0.5)]]

    result = {}
    all = np.zeros((3,))
    n = 0
    for k, v in res.items():
        v = np.array(v)
        s = np.sum(v, axis=0)
        m = np.mean(v, axis=0)
        result[k] = list(m)
        all += s
        n += len(v)
    all_mean = all / n
    result['all'] = list(all_mean)
    return result


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_mura(args):
    MURA_MEAN = [0.1524366]
    MURA_STD = [0.1807950]

    train_transform = transforms.Compose([
        transforms.RandomCrop(512, padding=args.padding),
        # transforms.ColorJitter(),
        transforms.RandomRotation(args.rotation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MURA_MEAN, MURA_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MURA_MEAN, MURA_STD),
    ])
    return train_transform, valid_transform

def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    if isinstance(model, nn.DataParallel):
        return np.sum(np.prod(v.size()) for v in model.module.model_parameters()) / 1e6
    else:
        return np.sum(np.prod(v.size()) for v in model.model_parameters()) / 1e6


def count_parameters_woaux_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


# Written by Yin Zheng
def draw_genotype(genotype, n_nodes, filename, concat=None):
    """

    :param genotype: 
    :param filename: 
    :return: 
    """
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("-2", fillcolor='darkseagreen2')
    g.node("-1", fillcolor='darkseagreen2')
    steps = n_nodes

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for op, source, target in genotype:
        if source == 0:
            u = "-2"
        elif source == 1:
            u = "-1"
        else:
            u = str(source - 2)
        v = str(target-2)
        op = 'null' if op == 'none' else op
        g.edge(u, v, label=op, fillcolor="gray")


    g.node("out", fillcolor='palegoldenrod')
    if concat is not None:
        for i in concat:
            if i-2>=0:
                g.edge(str(i-2), "out", fillcolor="gray")
    else:
        for i in range(steps):
            g.edge(str(i), "out", fillcolor="gray")

    g.render(filename, view=False)


def arch_to_genotype(arch_normal, arch_reduce, n_nodes, cell_type, normal_concat=None, reduce_concat=None):
    try:
        primitives = eval(cell_type)
    except:
        assert False, 'not supported op type %s' % (cell_type)

    gene_normal = [(primitives[op], f, t) for op, f, t in arch_normal]
    gene_reduce = [(primitives[op], f, t) for op, f, t in arch_reduce]
    if normal_concat is not None:
        _normal_concat = normal_concat
    else:
        _normal_concat = range(2, 2 + n_nodes)
    if reduce_concat is not None:
        _reduce_concat = reduce_concat
    else:
        _reduce_concat = range(2, 2 + n_nodes)
    genotype = Genotype(normal=gene_normal, normal_concat=_normal_concat,
                        reduce=gene_reduce, reduce_concat=_reduce_concat)
    return genotype


def infinite_get(data_iter, data_queue):
    try:
        data = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(data_queue)
        data = next(data_iter)
    return data, data_iter


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_variable(inputs, device, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    return out


def arch_to_string(arch):
    return ', '.join(["(op:%d,from:%d,to:%d)" % (o, f, t) for o, f, t in arch])


def get_index_item(inputs):
    if isinstance(inputs, torch.Tensor):
        inputs = int(inputs.item())
    return inputs


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def arch_to_matrix(arch):
    f_list = []
    t_list = []
    for _, f, t in arch:
        f_list.append(f)
        t_list.append(t)
    return np.array(f_list), np.array(t_list)


def parse_arch(arch, num_node):
    f_list, t_list = arch_to_matrix(arch)
    adj = sp.coo_matrix((np.ones(f_list.shape[0]), (t_list, f_list)),
                        shape=(num_node, num_node),
                        dtype=np.float32)
    adj = adj.multiply(adj>0)
    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def sum_normalize(input):
    return input/torch.sum(input, -1, keepdim=True)


def convert_output(n_nodes, prev_nodes, prev_ops):
    """

    :param n_nodes: number of nodes
    :param prev_nodes: vector, each element is the node ID, int64, in the range of [0,1,...,n_node]
    :param prev_ops: vector, each element is the op_id, int64, in the range [0,1,...,n_ops-1]
    :return: arch list, (op, f, t) is the elements
    """
    assert len(prev_nodes) == 2 * n_nodes
    assert len(prev_ops) == 2 * n_nodes
    arch_list = []
    for i in range(n_nodes):
        t_node = i + 2
        f1_node = prev_nodes[i * 2].item()
        f2_node = prev_nodes[i * 2 + 1].item()
        f1_op = prev_ops[i * 2].item()
        f2_op = prev_ops[i * 2 + 1].item()
        arch_list.append((f1_op, f1_node, t_node))
        arch_list.append((f2_op, f2_node, t_node))
    return arch_list


def translate_arch(arch, action, op_type='FULLY_CONCAT_PRIMITIVES'):
    try:
        COMPACT_PRIMITIVES = eval(op_type)
    except:
        assert False, 'not supported op type %s' % (op_type)
    arch_list = []
    for idx, (op, f, t) in enumerate(arch):
        pruner_op_name = TRANSFORM_PRIMITIVES[action[idx]]
        if pruner_op_name == 'null' or pruner_op_name == 'skip_connect':
            f_op = COMPACT_PRIMITIVES.index(pruner_op_name)
        elif pruner_op_name == 'same':
            f_op = op
        else:
            assert False, 'invalid type %s in TRANSFORM_PRIMITIVES' % pruner_op_name
        arch_list.append((f_op, f, t))
    return arch_list


def genotype_to_arch(genotype, op_type='FULLY_CONCAT_PRIMITIVES'):
    try:
        COMPACT_PRIMITIVES = eval(op_type)
    except:
        assert False, 'not supported op type %s' % (op_type)
    arch_normal = [(COMPACT_PRIMITIVES.index(op), f, t) for op, f, t in genotype.normal]
    arch_reduce = [(COMPACT_PRIMITIVES.index(op), f, t) for op, f, t in genotype.reduce]
    return arch_normal, arch_reduce
