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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random

# from torch.autograd import Variable
from evaluate_model import NetworkCIFAR as Network
from scheduler import CosineWithRestarts

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--ngpus', type=int, default=1, help='number of gpus')
# scheduler restart
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--T_mul', type=float, default=2.0, help='multiplier for cycle')
parser.add_argument('--T0', type=int, default=10, help='The maximum number of epochs within the first cycle')
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of epochs for warmup')
parser.add_argument('--no_bias_decay', action='store_true', default=False, help='no bias decay')

# new
parser.add_argument('--prefix', type=str, default='.', help='parent save path: /opt/ml/disk/ for seven')

args = parser.parse_args()

if "SEVEN_JOB_ID" in os.environ:
    args.save = '{}-MEW-EVAL-{}'.format(os.environ['SEVEN_JOB_ID'], time.strftime("%Y%m%d-%H%M%S"))
else:
    args.save = 'NAS-MEW-EVAL-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

args.save = os.path.join(args.prefix, args.save)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0+ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model

def warmup_update_lr(optimizer, epoch, init_lr, warmup_epochs):
    """
    update learning rate of optimizers
    """
    lr = init_lr * (epoch+1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    # model = model.to(device)
    model = dataparallel(model, args.ngpus)

    logging.info("param size = %fMB", utils.count_parameters_woaux_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay
    # )

    group_weight = []
    group_bias = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            group_bias.append(param)
        else:
            group_weight.append(param)
    assert len(list(model.parameters())) == len(group_weight) + len(group_bias)
    optimizer = torch.optim.SGD([
        {'params': group_weight},
        {'params': group_bias, 'weight_decay': 0 if args.no_bias_decay else args.weight_decay}
    ], lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    if args.scheduler == "naive_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
    elif args.scheduler == 'consine_restart':
        scheduler = CosineWithRestarts(
            optimizer, t_0=args.T0, eta_min=args.learning_rate_min, last_epoch=-1, factor=args.T_mul
        )
    else:
        assert False, "unsupported schudeler type: %s" % args.scheduler

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_update_lr(optimizer, epoch, args.learning_rate, args.warmup_epochs)
        else:
            scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        # if isinstance(model, nn.DataParallel):
        #     model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # else:
        #     model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, device, epoch)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, device)
        logging.info('valid_acc %f', valid_acc)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer, device, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits, logits_aux = model(input, args.drop_path_prob * epoch / args.epochs)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        print('%fM' % (model.total_flops / 1e6))
        assert False

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            logits, _ = model(input, args.drop_path_prob)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
