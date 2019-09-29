import torch
import argparse
from genotypes import PRIMITIVES, LSTM_PRIMITIVES, LOOSE_END_PRIMITIVES, NOT_LOOSE_END_PRIMITIVES, HAND_PRIMITIVES, PRUNER_PRIMITIVES, Genotype
import numpy as np
import random
import genotypes

parser = argparse.ArgumentParser("CompactNAS")
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--num_steps', type=int, default=4, help='edge hidden dimension')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

genotype = eval("genotypes.%s" % args.arch)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

arch_normal, arch_reduce = [], []

rand_list = []

i=0
for op, f, t in genotype.normal:
    if i%2==0:
        rand_list.append(int(torch.randint(0, t, (1,)).item()))
    i+=1

i = 0
for op, f, t in genotype.normal:
    if i % 2 != 0:
        arch_normal.append((op, f, t))
    else:
        if t==5:
            arch_normal.append((op, f, t))
        else:
            arch_normal.append((op, rand_list[i//2], t))
    i += 1

i = 0
for op, f, t in genotype.reduce:
    if i % 2 != 0:
        arch_reduce.append((op, f, t))
    else:
        if t==5:
            arch_reduce.append((op, f, t))
        else:
            arch_reduce.append((op, rand_list[i//2], t))
    i += 1


_normal_concat = genotype.normal_concat

_reduce_concat = genotype.reduce_concat


pruned_genotype = Genotype(normal=arch_normal, normal_concat=_normal_concat,
                    reduce=arch_reduce, reduce_concat=_reduce_concat)

print(genotype)
print(pruned_genotype)
