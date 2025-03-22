import argparse
import torch
from models.fsvg_clip import FSVG
from thop import profile
from utils.misc import NestedTensor


def get_args_parser():
    parser = argparse.ArgumentParser("CLIP-VG Args", add_help=False)
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument("--model", type=str, default="ViT-L/14", help="Name of model to be exploited.")
    parser.add_argument("--ce", default=True, help="If true, use gaussian blur augmentation")
    parser.add_argument("--ce_keep", default=0.7, type=float, help="image size")
    parser.add_argument("--imsize", default=224, type=int, help="image size")
    parser.add_argument("--device", default="cpu", help="device to use for training / testing")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CLIP-VG evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    model = OSVG(args)
    img = torch.randn(1, 3, args.imsize, args.imsize)
    img_mask = torch.randn(1, args.imsize, args.imsize)
    input_txt = torch.randint(0, 1, (1, 77))
    img_data = NestedTensor(img, img_mask)
    text_data = NestedTensor(input_txt, input_txt)
    flops, params = profile(model, inputs=(img_data, text_data, args.ce_keep))
    print(f"GFLOPs: {flops / 1e9:.3f} G")
    print(f"PARAMs: {params / 1e6:.3f} M")
