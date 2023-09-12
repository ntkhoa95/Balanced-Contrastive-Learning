import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust
import math
# from tensorboardX import SummaryWriter
from dataset.inat import INaturalist
from dataset.mosquito import MosquitoDataset
from dataset.imagenet import ImageNetLT
# from models import resnet_big, resnext
from models.model_pool import ModelwEmb
from models import resnext
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision
from utils import GaussianBlur, shot_acc
# from torch.models.tensorboard import SummaryWriter
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet', "mosquito"])
parser.add_argument('--data', default='/DATACENTER/raid5/zjg/imagenet', metavar='DIR')
parser.add_argument('--arch', default='resnext50')
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float, help='cross entropy loss weight')
parser.add_argument('--beta', default=0.35, type=float, help='supervised contrastive loss weight')
parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str, choices=['sim-sim', 'sim-rand', 'rand-rand'],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=1024, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--use_norm', default=True, type=bool,
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')


num_classes = 6

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, 'batchsize', 
         str(args.batch_size), 
         'temp', str(args.temp), 
         args.cl_views])
    os.makedirs(args.root_log, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    # ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = 1
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        model = resnext.BCLModel(name='resnet50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnext50':
        model = resnext.BCLModel(name='resnext50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif str(args.arch).startswith("convnext") or str(args.arch).startswith("maxvit"):
        feat_dim = 1024 if str(args.arch).startswith("convnext") else 768
        model = ModelwEmb(
            num_classes=num_classes,
            arch=args.arch,
            pretrained=True,
            feat_dim=feat_dim
        )
    else:
        raise NotImplementedError('This model is not supported')

    # if args.gpu is not None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # torch.cuda.set_device(args.gpu)
    model = model.to(device)
    # else:
    #     model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    ## update 
    if args.dataset == "imagenet":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        txt_train = f'dataset/ImageNet_LT/ImageNet_LT_train.txt' 
        txt_val = f'dataset/ImageNet_LT/ImageNet_LT_val.txt'
    elif args.dataset == "inat":
        normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
        txt_train = f'dataset/iNaturalist18/iNaturalist18_train.txt'
        txt_val = f'dataset/iNaturalist18/iNaturalist18_val.txt'
    elif args.dataset == "mosquito":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        data_dir = args.data

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randncls = [
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0., 0.0)
        ], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_randnclsstack = [
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0., 0.)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0., 0.)  # not strengthened
        ], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    if args.cl_views == 'sim-sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'sim-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'randstack-randstack':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'inat':
        val_dataset = INaturalist(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False,
        )
        train_dataset = INaturalist(
            root=args.data,
            txt=txt_train,
            transform=transform_train
        )
    elif args.dataset == "imagenet":
        val_dataset = ImageNetLT(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False)
        train_dataset = ImageNetLT(
            root=args.data,
            txt=txt_train,
            transform=transform_train)
    elif args.dataset == "mosquito":
        val_dataset = MosquitoDataset(
            root=os.path.join(data_dir, "val"),
            transform=val_transform,
            phase="val"
        )
        train_dataset = MosquitoDataset(
            root=os.path.join(data_dir, "train"),
            transform=transform_train,
            phase="train"
        )

    ## need consider this attribute
    class_to_idx = train_dataset.class_to_idx
    print(np.unique(train_dataset.targets, return_counts=True))
    cls_num_list = np.unique(train_dataset.targets, return_counts=True)[1].tolist()
    args.cls_num = len(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    tf_writer = None
    best_acc1 = 0.0
    best_many, best_med, best_few, best_class = 0.0, 0.0, 0.0, []

    if args.dataset == "imagenet" or args.dataset == "inat":
        txt_test = f'dataset/ImageNet_LT/ImageNet_LT_test.txt' if args.dataset == 'imagenet' \
            else f'dataset/iNaturalist18/iNaturalist18_val.txt'
        test_dataset = INaturalist(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False
        ) if args.dataset == 'inat' else ImageNetLT(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        tf_writer = None
        acc1, many, med, few, class_accs = validate(train_loader, test_loader, model, args, tf_writer)
        print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}, Class Prec@1: {}'.format(acc1,
                                                                                                many,
                                                                                                med,
                                                                                                few,
                                                                                                class_accs))

    elif args.dataset == "mosquito":
        val_dataset = MosquitoDataset(
            root=os.path.join(data_dir, "val"),
            transform=val_transform,
            phase="val"
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        tf_writer = None
        acc1, many, med, few, class_accs = validate(train_loader, val_loader, model, args, tf_writer)
        print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}, Class Prec@1: {}'.format(acc1,
                                                                                                many,
                                                                                                med,
                                                                                                few,
                                                                                                class_accs))
    return

def validate(train_loader, val_loader, model, args, flag='val'):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)
            feat_mlp, logits, centers = model(inputs)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

            batch_time.update(time.time() - end)

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, class_accs = shot_acc(preds, total_labels, train_loader,
                                                                acc_per_cls=True)
        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1, class_accs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
