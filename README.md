# NAT: Neural Architecture Transformer for Accurate and Compact Architectures

Pytorch implementation for “NAT: Neural Architecture Transformer for Accurate and Compact Architectures”.

## A Simple Demo of NAT

<p align="center">
    <img src="./imgs/different_transformation.gif" width="100%"/>
</p>


## Requirements
```
Python>=3.6, PyTorch==0.4.0, torchvision==0.2.1 graphviz=0.10.1 scipy=1.1.0 pygcn
```

Please follow the [guide](https://github.com/tkipf/pygcn) to install pygcn.

## Datasets
We consider two benchmark classification datsets, including CIFAR-10 and ImageNet.

CIFAR-10 can be automatically downloaded by torchvision.

ImageNet needs to be manually downloaded (preferably to a SSD) following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

## Training Method

Train NAT on CIFAR-10. We consider two kinds of architectures with their associated operation sets, namely loose-end architectures and fully-concat architectures.
```
python train_search.py --data $DATA_DIR$ --op_type $OP_TYPE$
```


## Inference Method

### 1. Put the input architectures in [genotypes.py](./genotypes.py) as follows

```
DARTS = Genotype(
    normal=[('sep_conv_3x3', 0, 2), ('sep_conv_3x3', 1, 2), ('sep_conv_3x3', 0, 3), ('sep_conv_3x3', 1, 3), ('sep_conv_3x3', 1, 4),
            ('skip_connect', 0, 4), ('skip_connect', 0, 5), ('dil_conv_3x3', 2, 5)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0, 2), ('max_pool_3x3', 1, 2), ('skip_connect', 2, 3), ('max_pool_3x3', 1, 3), ('max_pool_3x3', 0, 4),
            ('skip_connect', 2, 4), ('skip_connect', 2, 5), ('max_pool_3x3', 1, 5)], reduce_concat=[2, 3, 4, 5])
```


### 2. Feed an architecture into the transformer and obtain the transformed architecture

You can obtain the transformed architecture by taking an architecture as input, *e.g.*, --arch DARTS.  


```
python nas_compact_derive.py --data ./data --arch DARTS --model_path pretrained/fully_connect.pt
```

<p align="center">
<img src="./imgs/darts.jpg" alt="darts" width="80%">
</p>
<p align="center">
Figure: An example of architecture transformation..
</p>


## Architecture Visualization

You can visualize both the input and the transformed architectures by
```
python visualize.py some_arch
```
where `some_arch` should be replaced by any architecture in [genotypes.py](./genotypes.py).


##Evaluation Method


To evaluate the performance of different architectures, we train the models from scratch on CIFAR-10 and ImageNet. We release the evaluation code for both data sets as follows.

**CIFAR-10** ([evaluate_cifar.py](./evaluate_cifar.py))


**ImageNet** ([evaluate_imagenet.py](./evaluate_imagenet.py))

