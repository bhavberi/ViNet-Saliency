import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from collections import OrderedDict
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from datasets import build_mvva_dataset
from utils import *
import utils
from model import VideoSaliencyModel
from data.transforms import build_transforms
from data.ava import MVVA_AVADataLoader 
import wandb


parser = argparse.ArgumentParser('VideoMAE training and evaluation script for MVVA DATASET', add_help=False)
parser.add_argument('--batch_size', default=64, type=int)  #
parser.add_argument('--epochs', default=30, type=int)  #


# Model parameters
parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--tubelet_size', type=int, default=2)
parser.add_argument('--input_size', default=224, type=int,
                    help='videos input size')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                    help='Attention dropout rate (default: 0.)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
parser.add_argument('--model_ema', action='store_true', default=False)
parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
parser.add_argument('--roi_align' , default=False , type=bool)

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""")

parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--layer_decay', type=float, default=0.75)

parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',  #
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',  # no use
                    help='Color jitter factor (default: 0.4)')

# Finetuning params
parser.add_argument('--finetune', default='', help='finetune from checkpoint')
parser.add_argument('--model_key', default='model|module', type=str)
parser.add_argument('--model_prefix', default='', type=str)
parser.add_argument('--init_scale', default=0.001, type=float)  # head init
parser.add_argument('--use_checkpoint', action='store_true')
parser.set_defaults(use_checkpoint=False)
parser.add_argument('--use_mean_pooling', action='store_true')  # cls_token
parser.set_defaults(use_mean_pooling=True)
parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

# Dataset parameters
parser.add_argument('--nb_classes', default=80, type=int,  #
                    help='number of the classification types')
parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
parser.add_argument('--num_segments', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=16)
parser.add_argument('--sampling_rate', type=int, default=4)
parser.add_argument('--data_set', default='Kinetics-400',
                    choices=['mvva','Kinetics-400', 'SSV2', 'UCF101', 'HMDB51', 'image_folder', 'ava'],
                    type=str, help='dataset')  #
parser.add_argument('--output_dir', default='',  #
                    help='path where to save, empty for no saving')
parser.add_argument('--log_dir', default=None,  #
                    help='path where to tensorboard log')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
parser.set_defaults(auto_resume=True)

parser.add_argument('--save_ckpt', action='store_true')
parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
parser.set_defaults(save_ckpt=True)

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--dist_eval', action='store_true', default=False,  #
                    help='Enabling distributed evaluation')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

parser.add_argument('--enable_deepspeed', action='store_true', default=False)

# Loss function args
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=False, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--step_size',default=5, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=1.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--l1',default=False, type=bool)

# WandB params
parser.add_argument('--use_wandb',default=False, type=bool)
parser.add_argument('--wandb_project',default="vinet0", type=str)
parser.add_argument('--wandb_username',default="bhavberi", type=str)

args = parser.parse_args()
print(args)

if args.use_wandb:
    config = {
        "lr": args.lr, 
        "model_type": "VideoSaliency",
        "Backbone": "VideoMae",
        "criterion": "KLDivLoss",
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "wandb_run_name": "VideoMae Action Detection"
    }

    wandb.init(
        entity = args.wandb_username,   # wandb username
        project = args.wandb_project,   # wandb project name. New project will be created if given project is missing.
        config = config         # Config dict
        )
    wandb.run.name = "VideoMae Action Detection Run 1"

device = torch.device(args.device)

# Transforms
transform_train = build_transforms(is_train=True)

# TRAINING DATA SET
dataset_train = build_mvva_dataset(
    is_train=True, transforms=transform_train)
data_loader_train =MVVA_AVADataLoader(
    dataset_train,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,  # opt.val.get('workers', 1),
    pin_memory=False,
    sampler=None,
    drop_last=True
)
data_loader_train.num_samples = len(dataset_train)
print("Number of training samples: {}".format(len(dataset_train)))

# VALIDATION DATA SET
transform_test = build_transforms(is_train=False)
dataset_val = build_mvva_dataset(is_train=False, transforms=transform_test)
data_loader_val = MVVA_AVADataLoader(
    dataset_val,
    batch_size=args.batch_size,#args.batch_size,
    shuffle=False,
    num_workers=0,#opt.val.get('workers', 1),
    pin_memory=False,
    sampler=None,
    drop_last=True
)
data_loader_val.num_samples = len(dataset_val)
print("Number of validation samples: {}".format(len(dataset_val)))

# MODEL
model = VideoSaliencyModel(
    all_frames=args.num_frames * args.num_segments,
    tubelet_size=args.tubelet_size,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    attn_drop_rate=args.attn_drop_rate,
    use_checkpoint=args.use_checkpoint,
    use_mean_pooling=args.use_mean_pooling,
    init_scale=args.init_scale,
    batch_size = args.batch_size,
    roi_align = args.roi_align,
)
print("VideoMae Model with ViNet Decoder")
#print(model.parameters)

patch_size = model.backbone.patch_embed.patch_size  # 16
print("Patch size = %s" % str(patch_size))

# LOADING PRETRAINED MODEL
checkpoint = torch.load(args.finetune)
print("Load ckpt from %s" % args.finetune)
checkpoint_model = None
for model_key in args.model_key.split('|'):
    if model_key in checkpoint:
        checkpoint_model = checkpoint[model_key]
        print("Load state_dict by model_key = %s" % model_key)
        break
if checkpoint_model is None:
    checkpoint_model = checkpoint
state_dict = model.backbone.state_dict() 
for k in ['head.weight', 'head.bias']:  #
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

all_keys = list(checkpoint_model.keys())
new_dict = OrderedDict()
for key in all_keys:
    if key.startswith('backbone.'):
        new_dict[key[9:]] = checkpoint_model[key]
    elif key.startswith('encoder.'):
        new_dict[key[8:]] = checkpoint_model[key]
    else:
        new_dict[key] = checkpoint_model[key]
checkpoint_model = new_dict

    # interpolate position embedding
if 'pos_embed' in checkpoint_model:
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
    num_patches = model.patch_embed.num_patches  #
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

    # height (== width) for the checkpoint position embedding 
    orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                    args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size,orig_size, embedding_size)
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size,new_size, new_size, embedding_size)
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

utils.load_state_dict(model.backbone, checkpoint_model, prefix=args.model_prefix)

# OPTIMIZER
skip_weight_decay_list = model.backbone.no_weight_decay()
num_layers = model.backbone.get_num_layers()
if args.layer_decay < 1.0:
    assigner = LayerDecayValueAssigner(
        list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
else:
    assigner = None

optimizer = create_optimizer(
        args, model, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)


def validate(model,epoch, device, args):
    model.eval()
    # tic = time.time()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    # tic = time.time()
    
    for i, (samples, gt , boxes,_) in enumerate(data_loader_val):
        gt_last_frame_list =torch.stack( [batch[-1] for batch in gt] )
        samples = torch.stack(samples)
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                boxes[i][j] = torch.tensor(boxes[i][j], device=device)

        print("Ground Truth : " , gt_last_frame_list.shape)   
        outputs = model(samples, boxes)
        
        if(outputs.size(0) != gt_last_frame_list.size(0)):
            continue
        loss = loss_func(outputs , gt_last_frame_list , args)
        cc_loss = cc(outputs , gt_last_frame_list)
        sim_loss = similarity(outputs , gt_last_frame_list)
        
        total_loss.update(loss.item())
        total_sim_loss.update(sim_loss.item())
        total_cc_loss.update(cc_loss.item())
    
    # time_taken = (time.time()-tic)/60

    # print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f} sim_loss : {:.5f}, time : {:3f}'.format(epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg))

    # return total_loss.avg

    data_to_log = {
        'val_loss': total_loss.avg,
        'val_cc_loss': total_cc_loss.avg,
        'val_sim_loss': total_sim_loss.avg,
    }

    return data_to_log


for epoch in range(args.epochs):
    model.train(True)
    
    total_loss = AverageMeter()
    cur_loss = AverageMeter()

    for i, (samples, gt , boxes,_) in enumerate(data_loader_train):
        gt_last_frame_list =torch.stack( [batch[-1] for batch in gt])
        samples = torch.stack(samples)
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                boxes[i][j] = torch.tensor(boxes[i][j], device=device)

        optimizer.zero_grad()

        outputs = model(samples, boxes)
        print("Model Output: ",outputs.shape)        
        print("Ground Truth: ",gt_last_frame_list.size())
        if(outputs.size(0) != gt_last_frame_list.size(0)):
            continue
        outputs = outputs.view(gt_last_frame_list.size())
        loss = loss_func(outputs , gt_last_frame_list , args)
        total_loss.update(loss.item())
        cur_loss.update(loss.item())

        loss.backward()
        optimizer.step()
    
    # print("Epoch %d, Total Loss = %.4f, Cur Loss = %.4f" % (epoch , total_loss.avg , cur_loss.avg))

    data_to_log_train = {
        'epoch': epoch,
        'train loss': total_loss.avg,
    }

    with torch.no_grad():
        data_to_log_val = validate(model, epoch, device, args)
    
    data_to_log = {**data_to_log_train, **data_to_log_val}
    print(data_to_log)

    if args.use_wandb:
        wandb.log(data_to_log)

if(args.use_wandb):
    wandb.finish()
