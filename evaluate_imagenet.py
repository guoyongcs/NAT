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
import torchvision.transforms as transforms
import time

from torch.autograd import Variable
from evaluate_model import NetworkImageNet as Network
from scheduler import CosineWithRestarts
import torch

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--num_workers', type=int, default=24, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=2, help='test (go over all validset) frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
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
parser.add_argument('--prefix', type=str, default='.', help='parent save path: /opt/ml/disk/ for seven')
# scheduler restart
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler: naive_cosine | consine_restart | decay | step')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--T_mul', type=float, default=2.0, help='multiplier for cycle')
parser.add_argument('--T0', type=int, default=10, help='The maximum number of epochs within the first cycle')
# darts settings
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
# image size
parser.add_argument('--train_size', type=int, default=299, help='train image size')
parser.add_argument('--eval_size', type=int, default=331, help='eval image size')
parser.add_argument('--final_dropout', type=float, default=0.0, help='final dropout')
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of epochs for warmup')
parser.add_argument('--no_bias_decay', action='store_true', default=False, help='no bias decay')
# load checkpoint
parser.add_argument('--checkpoint', type=str, default='', help='path for saved checkpoint')
parser.add_argument('--last_epoch', type=int, default=-1, help='last epoch to begin')


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

IMAGENET_CLASSES = 1000

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



class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


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
  model = Network(args.init_channels, IMAGENET_CLASSES, args.layers, args.auxiliary, genotype, args.final_dropout)

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

  if args.checkpoint != '':
      logging.info('load checkpoint %s' % args.checkpoint)
      cpt = torch.load(args.checkpoint)
      args.last_epoch = cpt['epoch'] if args.last_epoch == -1 else args.last_epoch
      model.load_state_dict(cpt['state_dict'])
      best_acc_top1 = cpt['best_acc_top1']
      # try:
      #   optimizer.load_state_dict(cpt['optimizer'])
      # except:
      #   print('cannot load optimizer states')


  model = dataparallel(model, args.ngpus)

  logging.info("param size = %fMB", utils.count_parameters_woaux_in_MB(model.module if isinstance(model, nn.DataParallel) else model))


  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)
  criterion_smooth = CrossEntropyLabelSmooth(IMAGENET_CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.to(device)

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(args.train_size),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(args.eval_size + 32),
      transforms.CenterCrop(args.eval_size),
      transforms.ToTensor(),
      normalize,
    ]))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

  if args.scheduler == "naive_cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1
    )
  elif args.scheduler == 'consine_restart':
    scheduler = CosineWithRestarts(
        optimizer, t_0=args.T0, eta_min=args.learning_rate_min, factor=args.T_mul, last_epoch=-1
    )
  elif args.scheduler == 'decay':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.decay_period, gamma=args.gamma, last_epoch=-1
    )
  elif args.scheduler == 'step':
    '''
    lr = 0.1     if epoch < 1/3 * epochs
    lr = 0.01    if 1/3 * epochs <= epoch < 2/3 * epochs
    lr = 0.001   if epoch >= 2/3 * epochs
    '''
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.epochs//3, args.epochs//3*2], gamma=0.1, last_epoch=-1
    )
  else:
    assert False, "unsupported schudeler type: %s" % args.scheduler


  best_acc_top1 = 0

  for i in range(args.last_epoch - args.warmup_epochs):
      scheduler.step()

  for epoch in range(args.last_epoch if args.last_epoch > -1 else 0, args.warmup_epochs + args.epochs):
    if epoch < args.warmup_epochs:
      warmup_update_lr(optimizer, epoch, args.learning_rate, args.warmup_epochs)
    else:
      scheduler.step()

    logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, device, epoch)
    logging.info('train_acc %f', train_acc)

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, device)
    logging.info('valid_acc top1 %f, top5 %f', valid_acc_top1, valid_acc_top5)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True
    try:
      utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'best_acc_top1': best_acc_top1,
        'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)
    except:
      print('cannot save checkpoint')

def train(train_queue, model, criterion, optimizer, device, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    # start_time = time.time()
    # end_time = start_time
    input = input.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    logits, logits_aux = model(input, args.drop_path_prob * epoch / args.epochs)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    print('%fM' % (model.total_flops / 1e6))
    assert False
    # end_time = time.time()
    # iter_time = end_time - start_time
    # print(iter_time)
    if step % args.report_freq == 0:
      logging.info('train %03d [%03d/%03d] %e %f %f', epoch, step, len(train_queue), objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion, device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.to(device)
    target = target.to(device)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 