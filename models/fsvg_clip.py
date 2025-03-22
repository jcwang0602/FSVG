import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import *
from utils.misc import NestedTensor


def resize_pos_embed(clip_visual, patch_size, imsize_new):
    # for patch embedding
    cls_pos_embed = clip_visual.positional_embedding[:1, :]
    patch_pos_embed_old = clip_visual.positional_embedding[1:, :]
    patch_pos_embed_old = patch_pos_embed_old.transpose(0, 1)
    E, Q = patch_pos_embed_old.shape
    P_H, P_W = clip_visual.input_resolution // patch_size, clip_visual.input_resolution // patch_size
    patch_pos_embed_old = patch_pos_embed_old.view(E, P_H, P_W).unsqueeze(0)

    # for search region
    H, W = imsize_new, imsize_new
    new_P_H, new_P_W = H // patch_size, W // patch_size
    patch_pos_embed_new = nn.functional.interpolate(patch_pos_embed_old, size=(new_P_H, new_P_W), mode="bicubic", align_corners=False)
    patch_pos_embed_new = patch_pos_embed_new.flatten(2).transpose(1, 2).squeeze(0)
    patch_pos_embed_new = torch.cat([cls_pos_embed, patch_pos_embed_new], dim=0)
    patch_pos_embed_new = nn.Parameter(patch_pos_embed_new)

    # for cls token
    clip_visual.positional_embedding = patch_pos_embed_new

    return clip_visual


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor):
    lens_s = attn.shape[-1] - lens_t - 1  # 256

    lens_keep = math.ceil(keep_ratio * lens_s)  # 180
    if lens_keep == lens_s:
        # return tokens, global_index, None
        return tokens

    attn_t = attn[:, lens_s + 1 :, 1 : lens_s + 1]  # [32, 77, 256] 获取文本和图片的注意力特征图
    attn_t = attn_t.mean(dim=1)  # B, L-T, L_s --> B, L_s [32, 256]

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    # non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    # keep_index = global_index.gather(dim=1, index=topk_idx)
    # removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_cls = tokens[:, :1]  # [32, 1, 768]
    tokens_s = tokens[:, 1 : lens_s + 1]  # [32, 256, 768]
    tokens_t = tokens[:, lens_s + 1 :]  # [32, 77, 768]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # concatenate these tokens
    tokens_new = torch.cat([tokens_cls, attentive_tokens, tokens_t], dim=1)
    # return tokens_new, keep_index, removed_index
    return tokens_new


def ceblock_forward(block, x, lens_t, keep_ratio_search=None):
    x_attn, attn = block.attention(block.ln_1(x))
    x = x + x_attn  # [197, 32, 768]
    x = x.permute(1, 0, 2)
    x = candidate_elimination(attn, x, lens_t, keep_ratio_search, None)
    x = x.permute(1, 0, 2)

    x = x + block.mlp(block.ln_2(x))  # [197, 32, 768]
    # return x, global_index_template, global_index_search, removed_index_search, attn
    return x


class MultiLevel_Transformer(nn.Module):
    def __init__(self, clip_vit, extract_layer):
        super().__init__()
        heads = clip_vit.width // 64
        self.width = clip_vit.width
        self.layers = clip_vit.layers
        self.resblocks = clip_vit.resblocks
        self.extract_layer = extract_layer

    def forward(self, x: torch.Tensor, lens_t, ce_keep_rate, ce_loc):
        for i in range(max(self.extract_layer) + 1):
            if ce_keep_rate < 1 and i in ce_loc:
                x = ceblock_forward(self.resblocks[i], x, lens_t, ce_keep_rate)
            else:
                x = self.resblocks[i](x)
        return x


class MultiLevel_ImageEncoder_modified(nn.Module):
    def __init__(self, clip_visu_model, extract_layer):
        super().__init__()
        self.input_resolution = clip_visu_model.input_resolution
        self.output_dim = clip_visu_model.output_dim
        self.conv1 = clip_visu_model.conv1
        self.class_embedding = clip_visu_model.class_embedding
        self.positional_embedding = clip_visu_model.positional_embedding
        self.ln_pre = clip_visu_model.ln_pre
        self.transformer = MultiLevel_Transformer(clip_visu_model.transformer, extract_layer)
        self.ln_post = clip_visu_model.ln_post
        self.proj = clip_visu_model.proj
        self.positional_embedding.requires_grad_(True)

    def forward(self, x, text_tensors, ce_keep_rate, ce_loc):
        txt_len = text_tensors.shape[1]
        x = self.conv1(x)  # [32, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [32, 768, 196]
        x = x.permute(0, 2, 1)  # [32, 196, 768]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # [32, 197, 768]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)  # LN
        x = torch.cat([x, text_tensors], dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND, B L H -> L B H [197, 32, 768]
        x = self.transformer(x, txt_len, ce_keep_rate, ce_loc)
        cls_token = x[0, :, :]  # split txt features
        return cls_token


class TextEncoder_modified(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, x):
        x = self.token_embedding(x).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection

        return x


class FSVG(nn.Module):
    def __init__(self, args):
        super(FSVG, self).__init__()
        print("This is the OSVG model.")
        # CLIP Model name: ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if args.model == "ViT-L/14":
            print("init ViT-L/14")
            self.clip, _ = clip.load("ViT-L/14", device=args.device)
            self.extract_layer = [5, 11, 17, 23]
            self.ce_layer = [6, 12, 18]
            self.patch_size = 14
            re_in_dim = 768
            re_out_dim = 1024
        elif args.model == "ViT-L/14@336px":
            print("init ViT-L/14@336px")
            self.clip, _ = clip.load("ViT-L/14@336px", device=args.device)
            self.extract_layer = [5, 11, 17, 23]
            self.ce_layer = [6, 12, 18]
            self.patch_size = 14
            re_in_dim = 768
            re_out_dim = 1024
        elif args.model == "ViT-B/32":
            print("init ViT-B/32")
            self.clip, _ = clip.load("ViT-B/32", device=args.device)
            self.extract_layer = [2, 5, 8, 11]
            self.ce_layer = [3, 6, 9]
            self.patch_size = 32
            re_in_dim = 512
            re_out_dim = 768
        else:  # default
            print("init ViT-B/16")
            self.clip, _ = clip.load("ViT-B/16", device=args.device)
            self.extract_layer = [2, 5, 8, 11]
            self.ce_layer = [3, 6, 9]
            self.patch_size = 16
            re_in_dim = 512
            re_out_dim = 768

        self.text_resize = FeatureResizer(
            input_feat_size=re_in_dim,
            output_feat_size=re_out_dim,
            dropout=0.1,
        )

        # resize pos embed
        if self.clip.visual.input_resolution != args.imsize:
            resize_pos_embed(self.clip.visual, self.patch_size, args.imsize)
        self.image_encoder = MultiLevel_ImageEncoder_modified(self.clip.visual, self.extract_layer)
        self.text_encoder = TextEncoder_modified(self.clip)
        self.bbox_embed = MLP(256, 256, 4, 3)
        self.neck = nn.Linear(self.clip.visual.transformer.width, 256)

    def tensorize_inputs(self, images: NestedTensor, texts: NestedTensor):
        image_tensors = images.tensors
        texts_tensors = texts.tensors
        return image_tensors, texts_tensors

    def forward(self, img_data, text_data, ce_keep_rate):
        image_tensors, text_tensors = self.tensorize_inputs(img_data, text_data)
        text_features = self.text_encoder(text_tensors)  # B * 77 * 512
        visu_src = self.image_encoder(image_tensors.type(self.clip.dtype), self.text_resize(text_features), ce_keep_rate, self.ce_layer)  # B * 197 * 512
        visu_src = self.neck(visu_src)  # B 4H -> B H
        pred_box = self.bbox_embed(visu_src).sigmoid()
        return pred_box


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output
