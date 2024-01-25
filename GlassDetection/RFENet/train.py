"""
training code
also contains the F-score evaluation code.
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import numpy as np
import torch
from apex import amp

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist, cal_mae, cal_ber, set_bn_eval
from utils.f_boundary import eval_mask_boundary
import datasets
import loss
import network
import optimizer
import pandas as pd

parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str)
parser.add_argument('--dataset', type=str, default='Trans10k')

parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use.')
parser.add_argument('--class_uniform_pct', type=float, default=0.0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--dice_loss', default=False, action='store_true', help="whether use dice loss in edge")
parser.add_argument("--ohem", action="store_true", default=False, help="start OHEM loss")
parser.add_argument("--aux", action="store_true", default=False, help="whether use Aux loss")
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--edge_weight', type=float, default=0.25,
                    help='edge loss weight')
parser.add_argument('--middle_weight', type=float, default=0.01,
                    help='middle segmentation loss weight')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='final segmentation loss weight')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')
parser.add_argument('--apex', action='store_true', default=False,
                    help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                         '2 means hard negative mining iter 2')
parser.add_argument('--trunk', type=str, default='resnet101')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--eval_epoch', type=int, default=150, help="start evaluation epoch")
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=4,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=1.0,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=1.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--log_root', type=str, default='', help='the logging root path.')
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--fix_bn', action='store_true', default=False,
                    help=" whether to fix bn for improving the performance")
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                         'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--print_freq', type=int, default=5, help='frequency of print')
parser.add_argument('--eval_freq', type=int, default=1, help='frequency of evaluation during training')
parser.add_argument('--num_cascade', type=int, default=4, help='number of cascade layers')
parser.add_argument('--weight_mean', type=int, default=0)
parser.add_argument('--thicky', default=8, type=int)
parser.add_argument('--edge_num_points', type=int, default=96)
parser.add_argument('--region_num_points', type=int, default=96)
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--mlp_ratio', type=float, default=2.)
parser.add_argument('--with_f1', action='store_true', default=False)
parser.add_argument('--beta', default=1, type=float)

args = parser.parse_args()

args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'mean_iu_without_bkg': 0,
                    'mBer': 10000, 'mBer_without_bkg': 10000, 'mae': 10000,
                    'fwavacc': 0}

torch.backends.cudnn.benchmark = True
args.world_size = 1

if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ and args.apex:
    args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

if args.apex:
    torch.cuda.set_device(args.local_rank)
    print('My Rank:', args.local_rank)

    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')


def main():
    """
    Main Function
    """

    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    criterion, criterion_val = loss.get_loss(args)
    net = network.get_net(args, criterion)

    optim, scheduler = optimizer.get_optimizer(args, net)

    if args.fix_bn:
        net.apply(set_bn_eval)
        print("Fix bn for finetuning")

    if args.fp16:
        net, optim = amp.initialize(net, optim, opt_level="O1")

    net = network.wrap_network_in_dataparallel(net, args.apex)
    if args.snapshot:
        optimizer.load_weights(net, optim,
                               args.snapshot, args.restore_optimizer)

    torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.max_epoch):

        cfg.immutable(False)
        cfg.EPOCH = epoch
        cfg.immutable(True)

        scheduler.step()
        train(train_loader, net, optim, epoch, writer)
        if args.apex:
            train_loader.sampler.set_epoch(epoch + 1)
        if epoch % args.eval_freq == 0 or epoch == args.max_epoch - 1:
            validate(val_loader, net, criterion_val,
                     optim, epoch, writer)
        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                if args.apex:
                    train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()


def train(train_loader, net, optim, curr_epoch, writer):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: the network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_main_loss = AverageMeter()
    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        inputs, gts, _, edges, _ = data

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs, gts, edges = inputs.cuda(), gts.cuda(), edges.cuda()

        optim.zero_grad()

        main_loss_dic = net(inputs, gts=(gts, edges))
        main_loss = 0.0
        for v in main_loss_dic.values():
            main_loss = main_loss + v

        if args.apex:
            log_main_loss = main_loss.clone().detach_()
            torch.distributed.all_reduce(log_main_loss, torch.distributed.ReduceOp.SUM)
            log_main_loss = log_main_loss / args.world_size
        else:
            main_loss = main_loss.mean()
            log_main_loss = main_loss.clone().detach_()

        train_main_loss.update(log_main_loss.item(), batch_pixel_size)
        if args.fp16:
            with amp.scale_loss(main_loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            if not torch.isfinite(main_loss).all():
                raise FloatingPointError(
                    "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                        curr_iter, main_loss
                    )
                )
            main_loss.backward()

        optim.step()

        curr_iter += 1

        if args.local_rank == 0 and i % args.print_freq == 0:
            msg = f'[epoch {curr_epoch}], [iter {i + 1} / {len(train_loader)}], '
            temp_msg = '[[seg loss {:0.5f}], [seg_edge_loss {:0.5f}]]'.format(
                main_loss_dic['seg_loss'], main_loss_dic['edge_loss'])
            msg += temp_msg
            msg += ', [lr {:0.5f}]'.format(optim.param_groups[-1]['lr'])

            logging.info(msg)

            writer.add_scalar('training/loss', (train_main_loss.val),
                              curr_iter)
            writer.add_scalar('training/lr', optim.param_groups[-1]['lr'],
                              curr_iter)

        if i > 5 and args.test_mode:
            return


def validate(val_loader, net, criterion, optim, curr_epoch, writer):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()

    iou_acc = 0
    total_mae = []
    total_bers = np.zeros((args.dataset_cls.num_classes,), dtype=np.float)
    total_bers_count = np.zeros((args.dataset_cls.num_classes,), dtype=np.float)

    dump_images = []

    for val_idx, data in enumerate(val_loader):
        inputs, gt_image, img_names = data
        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs)

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == args.dataset_cls.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)
        predictions = output.data.max(1)[1].cpu()

        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             args.dataset_cls.num_classes)

        total_mae.append(cal_mae(predictions.numpy(), gt_image.squeeze().numpy()))
        temp_bers, temp_bers_count = cal_ber(predictions.numpy(), gt_image.squeeze().numpy(),
                                             args.dataset_cls.num_classes)
        total_bers += temp_bers
        total_bers_count += temp_bers_count

        del output, val_idx, data

    if args.apex:
        iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
        torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
        iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, val_loss, iou_acc, total_mae, total_bers, total_bers_count, dump_images,
                      writer, curr_epoch, args.dataset_cls)

    return val_loss.avg


if __name__ == '__main__':
    main()