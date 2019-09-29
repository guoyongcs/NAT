from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_multi = namedtuple('Genotype', 'normal_bottom normal_concat_bottom reduce_bottom reduce_concat_bottom \
                                         normal_mid normal_concat_mid reduce_mid reduce_concat_mid \
                                         normal_top normal_concat_top')

NOT_LOOSE_END_PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_3x3',
    'null'
]

LOOSE_END_PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'null'
]

TRANSFORM_PRIMITIVES = [
    'none',
    'same',
    'skip_connect',
]

DARTS = Genotype(
    normal=[('sep_conv_3x3', 0, 2), ('sep_conv_3x3', 1, 2), ('sep_conv_3x3', 0, 3), ('sep_conv_3x3', 1, 3), ('sep_conv_3x3', 1, 4),
            ('skip_connect', 0, 4), ('skip_connect', 0, 5), ('dil_conv_3x3', 2, 5)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0, 2), ('max_pool_3x3', 1, 2), ('skip_connect', 2, 3), ('max_pool_3x3', 1, 3), ('max_pool_3x3', 0, 4),
            ('skip_connect', 2, 4), ('skip_connect', 2, 5), ('max_pool_3x3', 1, 5)], reduce_concat=[2, 3, 4, 5])

NAT_DARTS = Genotype(
    normal=[('sep_conv_3x3', 0, 2), ('sep_conv_3x3', 1, 2), ('skip_connect', 0, 3), ('sep_conv_3x3', 1, 3), ('skip_connect', 1, 4), ('skip_connect', 0, 4), ('skip_connect', 0, 5), ('dil_conv_3x3', 2, 5)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0, 2), ('skip_connect', 1, 2), ('skip_connect', 2, 3), ('skip_connect', 1, 3), ('max_pool_3x3', 0, 4), ('null', 2, 4), ('skip_connect', 2, 5), ('max_pool_3x3', 1, 5)], reduce_concat=range(2, 6))

ENAS = Genotype(
    normal=[('sep_conv_3x3', 1, 2), ('skip_connect', 1, 2), ('skip_connect', 0, 3), ('sep_conv_5x5', 1, 3),
            ('avg_pool_3x3', 0, 4), ('sep_conv_3x3', 1, 4), ('sep_conv_3x3', 0, 5), ('avg_pool_3x3', 1, 5), ('avg_pool_3x3', 0, 6), ('sep_conv_5x5', 1, 6)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 0, 2), ('avg_pool_3x3', 1, 2), ('sep_conv_3x3', 1, 3), ('avg_pool_3x3', 1, 3),
            ('avg_pool_3x3', 1, 4), ('sep_conv_3x3', 1, 4), ('avg_pool_3x3', 1, 5), ('sep_conv_5x5', 4, 5), ('sep_conv_5x5', 0, 6), ('sep_conv_3x3', 5, 6)],
    reduce_concat=[2, 3, 6])

NAT_ENAS = Genotype(
    normal=[('sep_conv_3x3', 1, 2), ('skip_connect', 1, 2), ('skip_connect', 0, 3), ('sep_conv_5x5', 1, 3), ('null', 0, 4), ('sep_conv_3x3', 1, 4), ('sep_conv_3x3', 0, 5), ('avg_pool_3x3', 1, 5), ('avg_pool_3x3', 0, 6), ('sep_conv_5x5', 1, 6)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[('sep_conv_5x5', 0, 2), ('avg_pool_3x3', 1, 2), ('sep_conv_3x3', 1, 3), ('avg_pool_3x3', 1, 3), ('avg_pool_3x3', 1, 4), ('sep_conv_3x3', 1, 4), ('null', 1, 5), ('sep_conv_5x5', 4, 5), ('sep_conv_5x5', 0, 6), ('sep_conv_3x3', 5, 6)],
    reduce_concat=[2, 3, 6])

# We make the normal cell and the reduction cell the same for VGG and ResBlock.
ResBlock = Genotype(
    normal=[('null', 0, 2), ('skip_connect', 1, 2), ('null', 0, 3), ('conv_3x3', 2, 3), ('null', 0, 4), ('conv_3x3', 3, 4), ('skip_connect', 2, 5), ('skip_connect', 4, 5)],
    normal_concat=[5],
    reduce=[('null', 0, 2), ('skip_connect', 1, 2), ('null', 0, 3), ('conv_3x3', 2, 3), ('null', 0, 4), ('conv_3x3', 3, 4), ('skip_connect', 2, 5), ('skip_connect', 4, 5)],
    reduce_concat=[5])

NAT_ResBlock = Genotype(
    normal=[('null', 0, 2), ('skip_connect', 1, 2), ('skip_connect', 0, 3), ('conv_3x3', 2, 3), ('skip_connect', 0, 4), ('conv_3x3', 3, 4), ('skip_connect', 2, 5), ('skip_connect', 4, 5)],
    normal_concat=[5],
    reduce=[('null', 0, 2), ('skip_connect', 1, 2), ('skip_connect', 0, 3), ('conv_3x3', 2, 3), ('skip_connect', 0, 4), ('conv_3x3', 3, 4), ('skip_connect', 2, 5), ('skip_connect', 4, 5)],
    reduce_concat=[5])

VGG = Genotype(
    normal=[('null', 0, 2), ('skip_connect', 1, 2), ('null', 0, 3), ('conv_3x3', 2, 3), ('null', 0, 4), ('conv_3x3', 3, 4), ('null', 0, 5), ('skip_connect', 4, 5)],
    normal_concat=[5],
    reduce=[('null', 0, 2), ('skip_connect', 1, 2), ('null', 0, 3), ('conv_3x3', 2, 3), ('null', 0, 4), ('conv_3x3', 3, 4), ('null', 0, 5), ('max_pool_3x3', 4, 5)],
    reduce_concat=[5])

NAT_VGG = Genotype(
    normal=[('skip_connect', 0, 2), ('skip_connect', 1, 2), ('null', 0, 3), ('conv_3x3', 2, 3), ('null', 0, 4), ('conv_3x3', 3, 4), ('skip_connect', 0, 5), ('skip_connect', 4, 5)],
    normal_concat=[5],
    reduce=[('skip_connect', 0, 2), ('skip_connect', 1, 2), ('null', 0, 3), ('conv_3x3', 2, 3), ('null', 0, 4),
            ('conv_3x3', 3, 4), ('skip_connect', 0, 5), ('skip_connect', 4, 5)],
    reduce_concat=[5])

# Results of some random architectures.
R1 = Genotype(normal=[('dil_conv_5x5', 1, 2), ('dil_conv_5x5', 1, 2), ('dil_conv_5x5', 1, 3), ('dil_conv_5x5', 2, 3), ('sep_conv_3x3', 0, 4), ('sep_conv_3x3', 3, 4), ('skip_connect', 1, 5), ('max_pool_3x3', 0, 5)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0, 2), ('max_pool_3x3', 0, 2), ('skip_connect', 0, 3), ('sep_conv_3x3', 0, 3), ('conv_3x3', 0, 4), ('max_pool_3x3', 2, 4), ('avg_pool_3x3', 2, 5), ('conv_3x3', 3, 5)], reduce_concat=range(2, 6))

NAT_R1 = Genotype(normal=[('skip_connect', 1, 2), ('dil_conv_5x5', 1, 2), ('null', 1, 3), ('dil_conv_5x5', 2, 3), ('sep_conv_3x3', 0, 4), ('sep_conv_3x3', 3, 4), ('skip_connect', 1, 5), ('max_pool_3x3', 0, 5)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0, 2), ('max_pool_3x3', 0, 2), ('skip_connect', 0, 3), ('sep_conv_3x3', 0, 3), ('skip_connect', 0, 4), ('skip_connect', 2, 4), ('skip_connect', 2, 5), ('conv_3x3', 3, 5)], reduce_concat=range(2, 6))

R2 = Genotype(normal=[('max_pool_3x3', 1, 2), ('dil_conv_3x3', 0, 2), ('sep_conv_5x5', 2, 3), ('avg_pool_3x3', 2, 3), ('dil_conv_5x5', 0, 4), ('skip_connect', 3, 4), ('sep_conv_3x3', 1, 5), ('max_pool_3x3', 3, 5)], normal_concat=range(2, 6), reduce=[('conv_3x3', 1, 2), ('avg_pool_3x3', 1, 2), ('max_pool_3x3', 0, 3), ('dil_conv_5x5', 0, 3), ('max_pool_3x3', 0, 4), ('skip_connect', 0, 4), ('dil_conv_5x5', 4, 5), ('skip_connect', 4, 5)], reduce_concat=range(2, 6))

NAT_R2 = Genotype(normal=[('max_pool_3x3', 1, 2), ('dil_conv_3x3', 0, 2), ('null', 2, 3), ('null', 2, 3), ('dil_conv_5x5', 0, 4), ('null', 3, 4), ('sep_conv_3x3', 1, 5), ('max_pool_3x3', 3, 5)], normal_concat=range(2, 6), reduce=[('skip_connect', 1, 2), ('avg_pool_3x3', 1, 2), ('skip_connect', 0, 3), ('dil_conv_5x5', 0, 3), ('null', 0, 4), ('skip_connect', 0, 4), ('dil_conv_5x5', 4, 5), ('skip_connect', 4, 5)], reduce_concat=range(2, 6))

R3 = Genotype(normal=[('conv_3x3', 0, 2), ('skip_connect', 0, 2), ('conv_3x3', 0, 3), ('max_pool_3x3', 0, 3), ('sep_conv_5x5', 1, 4), ('sep_conv_3x3', 3, 4), ('avg_pool_3x3', 4, 5), ('avg_pool_3x3', 4, 5)], normal_concat=range(2, 6), reduce=[('skip_connect', 0, 2), ('max_pool_3x3', 0, 2), ('sep_conv_5x5', 2, 3), ('avg_pool_3x3', 0, 3), ('skip_connect', 3, 4), ('max_pool_3x3', 2, 4), ('max_pool_3x3', 1, 5), ('sep_conv_5x5', 4, 5)], reduce_concat=range(2, 6))

NAT_R3 = Genotype(normal=[('conv_3x3', 0, 2), ('skip_connect', 0, 2), ('null', 0, 3), ('max_pool_3x3', 0, 3), ('null', 1, 4), ('null', 3, 4), ('avg_pool_3x3', 4, 5), ('null', 4, 5)], normal_concat=range(2, 6), reduce=[('skip_connect', 0, 2), ('max_pool_3x3', 0, 2), ('sep_conv_5x5', 2, 3), ('avg_pool_3x3', 0, 3), ('null', 3, 4), ('skip_connect', 2, 4), ('max_pool_3x3', 1, 5), ('sep_conv_5x5', 4, 5)], reduce_concat=range(2, 6))

R4 = Genotype(normal=[('conv_3x3', 1, 2), ('sep_conv_3x3', 1, 2), ('conv_3x3', 2, 3), ('conv_3x3', 2, 3), ('sep_conv_3x3', 3, 4), ('conv_3x3', 0, 4), ('max_pool_3x3', 0, 5), ('max_pool_3x3', 4, 5)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0, 2), ('sep_conv_5x5', 1, 2), ('skip_connect', 2, 3), ('sep_conv_3x3', 2, 3), ('sep_conv_5x5', 2, 4), ('conv_3x3', 1, 4), ('dil_conv_5x5', 2, 5), ('max_pool_3x3', 4, 5)], reduce_concat=range(2, 6))

NAT_R4 = Genotype(normal=[('conv_3x3', 1, 2), ('sep_conv_3x3', 1, 2), ('skip_connect', 2, 3), ('null', 2, 3), ('sep_conv_3x3', 3, 4), ('conv_3x3', 0, 4), ('max_pool_3x3', 0, 5), ('null', 4, 5)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0, 2), ('sep_conv_5x5', 1, 2), ('skip_connect', 2, 3), ('sep_conv_3x3', 2, 3), ('skip_connect', 2, 4), ('conv_3x3', 1, 4), ('null', 2, 5), ('skip_connect', 4, 5)], reduce_concat=range(2, 6))

R5 = Genotype(normal=[('skip_connect', 0, 2), ('max_pool_3x3', 0, 2), ('max_pool_3x3', 2, 3), ('dil_conv_5x5', 0, 3), ('max_pool_3x3', 2, 4), ('conv_3x3', 2, 4), ('conv_3x3', 0, 5), ('sep_conv_5x5', 2, 5)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0, 2), ('avg_pool_3x3', 1, 2), ('sep_conv_5x5', 0, 3), ('avg_pool_3x3', 2, 3), ('conv_3x3', 0, 4), ('dil_conv_5x5', 0, 4), ('sep_conv_3x3', 0, 5), ('skip_connect', 4, 5)], reduce_concat=range(2, 6))

NAT_R5 = Genotype(normal=[('skip_connect', 0, 2), ('max_pool_3x3', 0, 2), ('null', 2, 3), ('dil_conv_5x5', 0, 3), ('null', 2, 4), ('conv_3x3', 2, 4), ('conv_3x3', 0, 5), ('sep_conv_5x5', 2, 5)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0, 2), ('avg_pool_3x3', 1, 2), ('sep_conv_5x5', 0, 3), ('avg_pool_3x3', 2, 3), ('conv_3x3', 0, 4), ('null', 0, 4), ('skip_connect', 0, 5), ('skip_connect', 4, 5)], reduce_concat=range(2, 6))

