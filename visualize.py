import sys
from utils import draw_genotype
import numpy as np


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    draw_genotype(genotype.normal, np.max(list(genotype.normal_concat) + list(genotype.reduce_concat)) -1, "normal", genotype.normal_concat)
    draw_genotype(genotype.reduce, np.max(list(genotype.normal_concat) + list(genotype.reduce_concat)) -1, "reduction", genotype.reduce_concat)
