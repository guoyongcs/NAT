import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from graphviz import Digraph

from search_model import NASNetwork as Network
from nas_compact_architect import Architect
from nat_learner import Transformer
import random

import genotypes


parser = argparse.ArgumentParser("CompactNAS")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=2, help='test (go over all validset) frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='number of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--train_portion', nargs='?', default='[0.4,0.8]', help='portion of training data')
parser.add_argument('--train_portion', type=float, default=0.4, help='data portion for training weights')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch masters')
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--inner_steps', type=int, default=3, help='number of inner updates')
parser.add_argument('--inner_lr', type=float, default=0.001, help='learning rate for inner updates')
parser.add_argument('--valid_inner_steps', type=int, default=3, help='number of inner updates for validation')
parser.add_argument('--n_archs', type=int, default=10, help='number of candidate archs')
parser.add_argument('--prefix', type=str, default='.', help='parent save path: /opt/ml/disk/ for seven')
# lstm
parser.add_argument('--controller_type', type=str, default='SAMPLE', help='SAMPLE | LSTM')
parser.add_argument('--controller_hid', type=int, default=100, help='temperature for lstm')
parser.add_argument('--controller_temperature', type=float, default=None, help='temperature for lstm')
parser.add_argument('--controller_tanh_constant', type=float, default=None, help='tanh constant for lstm')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.005, 0.005], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of layers in lstm')
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5, help='coefficient for entropy')
# controller warmup
parser.add_argument('--controller_start_training', type=int, default=0, help='Epoch that the training of controller starts')
# scheduler restart
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--T_mul', type=float, default=2.0, help='multiplier for cycle')
parser.add_argument('--T0', type=int, default=10, help='The maximum number of epochs within the first cycle')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
# parser.add_argument('--benchmark_path', type=str, default=None, help='Path to restore the benchmark model')
# parser.add_argument('--restore_path', type=str, default=None, help='Path to restore the model')
# pruner
parser.add_argument('--pruner_learning_rate', type=float, default=3e-4, help='learning rate for pruner')
parser.add_argument('--pruner_weight_decay', type=float, default=5e-4, help='learning rate for pruner')
parser.add_argument('--edge_hid', type=int, default=100, help='edge hidden dimension')
parser.add_argument('--pruner_nfeat', type=int, default=1024, help='feature dimension of each node')
parser.add_argument('--pruner_nhid', type=int, default=100, help='hidden dimension')
parser.add_argument('--pruner_dropout', type=float, default=0, help='dropout rate for pruner')
parser.add_argument('--pruner_normalize', action='store_true', default=False, help='use normalize in GCN')
parser.add_argument('--loose_end', action='store_true', default=False, help='loose_end')
parser.add_argument('--split_fc', action='store_true', default=False, help='split_fc')

parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--num_steps', type=int, default=4, help='edge hidden dimension')
parser.add_argument('--op_type', type=str, default='NOT_LOOSE_END_PRIMITIVES', help='LOOSE_END_PRIMITIVES | NOT_LOOSE_END_PRIMITIVES | HAND_PRIMITIVES')

args = parser.parse_args()
if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

# args.train_portion = eval(args.train_portion)
if "SEVEN_JOB_ID" in os.environ:
    args.save = '{}-MEW-search-{}'.format(os.environ['SEVEN_JOB_ID'], time.strftime("%Y%m%d-%H%M%S"))
else:
    args.save = 'NAS-MEW-search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = os.path.join(args.prefix, args.save)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

CIFAR_CLASSES = 10


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info('GPU device = %d' % args.gpu)
    else:
        logging.info('no GPU available, use CPU!!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    genotype = eval("genotypes.%s" % args.arch)
    arch_normal, arch_reduce = utils.genotype_to_arch(genotype, args.op_type)

    model = Network(
        args.init_channels, CIFAR_CLASSES, args.layers, criterion, device, steps=args.num_steps,controller_type=args.controller_type, controller_hid=args.controller_hid, controller_temperature=args.controller_temperature, controller_tanh_constant=args.controller_tanh_constant, controller_op_tanh_reduce=args.controller_op_tanh_reduce, entropy_coeff=args.entropy_coeff, edge_hid = args.edge_hid, pruner_nfeat = args.pruner_nfeat, pruner_nhid = args.pruner_nhid, pruner_dropout = args.pruner_dropout, pruner_normalize = args.pruner_normalize, loose_end = args.loose_end, split_fc=args.split_fc, normal_concat=genotype.normal_concat, reduce_concat=genotype.reduce_concat, op_type=args.op_type
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)

    derive_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        shuffle=False, pin_memory=True, num_workers=2
    )

    model.load_state_dict(torch.load(args.model_path), strict=False)

    model.to(device)

    # derive pruned arch
    model.derive_pruned_arch(derive_queue, arch_normal, arch_reduce, 10, logger, args.save, "derive", normal_concat=genotype.normal_concat, reduce_concat=genotype.reduce_concat)

if __name__ == '__main__':
    main()
