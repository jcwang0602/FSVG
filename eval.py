import time
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
from engine import evaluate


def get_args_parser():
    parser = argparse.ArgumentParser("CLIP-VG Args", add_help=False)
    parser.add_argument("--sup_type", default="full", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--eval", dest="eval", default=False, action="store_true", help="if evaluation only")
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument("--model", type=str, default="ViT-L/14", help="Name of model to be exploited.")
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
    """ embedding size"""
    parser.add_argument("--emb_size", default=512, type=int, help="fusion module embedding dimensions")

    # Dataset parameters
    parser.add_argument("--data_root", type=str, default="image_data", help="path to ReferIt splits data folder")
    parser.add_argument("--split_root", type=str, default="split_data", help="location of pre-parsed dataset info")
    parser.add_argument("--dataset", default="unc", type=str, help="referit/unc/unc+/gref/gref_umd")
    parser.add_argument("--max_query_len", default=50, type=int, help="maximum time steps (lang length) per batch")
    parser.add_argument("--prompt", type=str, default="{pseudo_query}", help="Prompt template")
    # dataset parameters
    parser.add_argument("--output_dir", default="outputs/osvg_clip_vitb16_256_g2_b64_b5_g2_unc", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--retrain", default="", help="retrain from checkpoint")
    parser.add_argument("--light", dest="light", default=False, action="store_true", help="if use smaller model")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=4, type=int)
    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    # evalutaion options
    parser.add_argument("--eval_set", default="val", type=str)  # 'testA', 'testB', 'val'
    parser.add_argument("--eval_model", default="outputs/osvg_clip_vitb16_256_g2_b64_b5_g2_unc/best_checkpoint.pth", type=str)

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

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(
        dataset_test,
        args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.eval_model, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])
    print("Current model training epoch is: ", checkpoint["epoch"])

    # output log
    eval_model = args.eval_model
    eval_model = eval_model.split("/")[-1].split(".")[0]
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
            f.write(str(args) + "\n")
            f.flush()
    start_time = time.time()

    # perform evaluation
    accuracy = evaluate(args, model, data_loader_test, device)

    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Testing time {}".format(total_time_str))
        log_stats = {
            "test_model:": args.eval_model,
            "%s_set_accuracy" % args.eval_set: accuracy,
        }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CLIP-VG evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
