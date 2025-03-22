import os
import time
import math
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
import utils.misc as utils
from models.fsvg_clip import FSVG
from datasets import build_dataset
from engine import train_one_epoch, validate
from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser("CLIP-VG Args", add_help=False)
    parser.add_argument("--sup_type", default="full", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_text", default=1e-5, type=float)
    parser.add_argument("--lr_visu", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--lr_power", default=0.9, type=float, help="lr poly power")
    parser.add_argument("--lr_exponential", default=0.9, type=float, help="lr exponential")
    parser.add_argument("--clip_max_norm", default=0.0, type=float, help="gradient clipping max norm")
    parser.add_argument("--eval", dest="eval", default=False, action="store_true", help="if evaluation only")
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--lr_scheduler", default="step", type=str)
    parser.add_argument("--lr_drop", default=60, type=int)
    # Augmentation options
    parser.add_argument("--aug_blur", action="store_true", help="If true, use gaussian blur augmentation")
    parser.add_argument("--aug_crop", action="store_true", help="If true, use random crop augmentation")
    parser.add_argument("--aug_scale", action="store_true", help="If true, use multi-scale augmentation")
    parser.add_argument("--aug_translate", action="store_true", help="If true, use random translate augmentation")
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument("--model", type=str, default="ViT-L/14@336px", help="Name of model to be exploited.")
    parser.add_argument("--fs", action="store_true", help="If true, use gaussian blur augmentation")
    parser.add_argument("--fs_keep", default=0.7, type=float, help="image size")
    parser.add_argument("--fs_start", default=20, type=int, help="image size")
    parser.add_argument("--fs_warm", default=30, type=int, help="image size")
    parser.add_argument("--dim_feedforward", default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument("--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--imsize", default=384, type=int, help="image size")
    # Loss options
    parser.add_argument("--bbox_w", default=5.0, type=float, help="image size")
    parser.add_argument("--giou_w", default=2.0, type=float, help="image size")
    """ embedding size"""
    parser.add_argument("--emb_size", default=512, type=int, help="fusion module embedding dimensions")
    # Dataset parameters
    parser.add_argument("--data_root", type=str, default="image_data/", help="path to ReferIt splits data folder")
    parser.add_argument("--split_root", type=str, default="split_data/", help="location of pre-parsed dataset info")
    parser.add_argument("--dataset", default="referit", type=str, help="referit/unc/unc+/gref/gref_umd")
    parser.add_argument("--max_query_len", default=50, type=int, help="maximum time steps (lang length) per batch")
    # Prompt Engineering: "{pseudo_query}" denote without using prompt
    #                    "{pseudo_query}" or using "find the region that corresponds to the description {pseudo_query}"
    parser.add_argument("--prompt", type=str, default="{pseudo_query}", help="Prompt template")
    # dataset parameters
    parser.add_argument("--output_dir", default="./outputs/test", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--light", dest="light", default=False, action="store_true", help="if use smaller model")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=8, type=int)
    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    return parser


def main(args):
    """distribution init"""
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.model == "ViT-L/14" or args.model == "ViT-L/14@336px":
        args.vl_hidden_dim = 768
    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("### INFO ### torch.backends.cudnn.benchmark = {}".format(torch.backends.cudnn.benchmark))
    writer = SummaryWriter(args.output_dir)
    # build model
    model = FSVG(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print("number of requires_grad params: ", n_parameters_grad)
    print("number of all params: ", n_parameters)

    visu_param = []
    text_param = []
    rest_param = []

    visu_names = []
    text_names = []
    rest_names = []
    for n, p in model_without_ddp.named_parameters():
        if (("clip" in n and "visual" in n) or "text_resize" in n) and p.requires_grad:
            visu_param.append(p)
            visu_names.append(n)
        elif ("clip" in n) and ("transformer" in n or "ln_final" in n or "token_embedding" in n or "text_projection" in n or "positional_embedding" in n) and p.requires_grad:
            text_param.append(p)
            text_names.append(n)
        else:
            rest_param.append(p)
            rest_names.append(n)

    param_list = [{"params": rest_param, "lr": args.lr}, {"params": visu_param, "lr": args.lr_visu}, {"params": text_param, "lr": args.lr_text}]
    # using RMSProp or AdamW
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError("Lr scheduler type not supportted ")

    # using polynomial lr scheduler or half decay every 10 epochs or step
    if args.lr_scheduler == "poly":
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == "halfdecay":
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == "cosine":
        lr_func = lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == "exponential":
        lr_func = lambda epoch: args.lr_exponential**epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError("Lr scheduler type not supportted ")

    # build dataset
    print("build dataset...")
    if args.sup_type == "full":
        print("perform fullly supervised setting.")
        dataset_train = build_dataset("train", args)
    else:  # un
        print("perform unsupervised setting.")
        dataset_train = build_dataset("train_pseudo", args)

    # note certain dataset does not have 'test' set: eg. 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    dataset_val = build_dataset("val", args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    best_accu = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
            val_stats = validate(args, model, data_loader_val, device, args.start_epoch)
            best_accu = val_stats["accu"]
            print("best_accu: {}".format(best_accu))
        except Exception as e:
            print(e)
            print("Not founed the checkpoint!!!")

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(str(args) + "\n")

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        val_stats = validate(args, model, data_loader_val, device, epoch)
        log_stats = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}, "n_parameters": n_parameters}
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [os.path.join(args.output_dir, "checkpoint.pth")]
            if val_stats["accu"] > best_accu:
                checkpoint_paths.append(os.path.join(args.output_dir, "best_checkpoint.pth"))
                best_accu = val_stats["accu"]

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({"model": model_without_ddp.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict(), "epoch": epoch, "args": args, "val_accu": val_stats["accu"]}, checkpoint_path)
        if utils.is_main_process:
            writer.add_scalar("train/train_loss", train_stats["loss"], epoch)
            writer.add_scalar("train/bbox_loss", train_stats["loss_bbox"], epoch)
            writer.add_scalar("train/giou_loss", train_stats["loss_giou"], epoch)
            writer.add_scalar("val/miou", val_stats["miou"], epoch)
            writer.add_scalar("val/accu", val_stats["accu"], epoch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CLIP-VG training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
