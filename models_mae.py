# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed    #, Block
import numpy as np
from util.pos_embed import get_2d_sincos_pos_embed

import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from torch.jit import Final
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, SwiGLU, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias = attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias = attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # print(np.shape(attn_weight))
        # print(np.shape(attn_bias))

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        # print(np.shape(attn_weight))
        # print("Mean row sum:", attn_weight.sum(dim=-1).mean().item())
        attn_weight_1 = torch.dropout(attn_weight, dropout_p, train=True)
        # print("Mean row sum_1:", attn_weight_1.sum(dim=-1).mean().item())
        return attn_weight_1 @ value, attn_weight

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attention=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = None

        if self.fused_attn:
            x, attn = scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        # else:
        #     q = q * self.scale 
        #     attn = q @ k.transpose(-2, -1)      # Compute raw attention scores
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn, q, k, v  # Return both output and attention weights
        return x 

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, return_attention=False):
        if return_attention:
            attn_output, attn_weights, q, k, v = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path1(self.ls1(attn_output))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, attn_weights, q, k, v
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    #     x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    #     return x
# -----------------------------------------------------------------------------------------------------------------------------

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)      #removed qk_scale ...AS
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)  #removed qk_scale ...AS
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_attention=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = []
        attention_maps = []
        query = []
        key = []
        value = []

        for blk in self.blocks:
            if return_attention:
                x, attn_weights, q, k, v = blk(x, return_attention=True)
                attention_maps.append(attn_weights)
                query.append(q.clone())
                key.append(k.clone())
                value.append(v.clone())
            else:
                x = blk(x)
            embeddings.append(x.clone())    # layer_embeddings extract from here therefore, clustering good for mean patch (always stays unnormalized) 
                                            # also, similarity b/w 10 times run embeddings also same (irrespective of below normalization)

        x = self.norm(x)      # unnormalized me clustering good
        # Normalized can be used when checking similarity b/w 10 times run embeddings (cls or mean patch) -- same

        # cls_token_output = x[:, 0]

        if return_attention:
            return x, mask, ids_restore, embeddings, attention_maps, query, key, value
        return x, mask, ids_restore, embeddings

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # print(x.shape)         # (1,50,512)
        # print(ids_restore.shape)  # (1, 196)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)   # (b, 147, 1) masked patches

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    ## main one
    def forward(self, imgs, mask_ratio=0.75, return_attention=False):
        if return_attention:
            latent, mask, ids_restore, embeddings, attention_maps, query, key, value = self.forward_encoder(imgs, mask_ratio, return_attention=True)
        else:
            latent, mask, ids_restore, embeddings = self.forward_encoder(imgs, mask_ratio)

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        if return_attention:
            return loss, pred, mask, latent, embeddings, attention_maps, query, key, value  # extract attention maps
        return loss, pred, mask, latent, embeddings
    
        ## latent, mask, ids_restore, embeddings = self.forward_encoder(imgs, mask_ratio)
        ## pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        ## loss = self.forward_loss(imgs, pred, mask)
        ## return loss, pred, mask, latent, embeddings
    
    ## mean patch instead of cls token
    # def forward(self, imgs, mask_ratio=0.75):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     latent_org = latent.clone()
    #     mean_patch_embeddings = torch.mean(latent[:, 1:, :], dim=1)
    #     # Replace the class token (index 0) with the mean patch embedding
    #     latent[:, 0, :] = mean_patch_embeddings
    #     pred1 = self.forward_decoder(latent_org, ids_restore)  # class token
    #     pred2 = self.forward_decoder(latent, ids_restore)  # mean patch token
    #     loss1 = self.forward_loss(imgs, pred1, mask)
    #     loss2 = self.forward_loss(imgs, pred2, mask)
    #     return loss1, loss2, pred1, pred2, mask, latent_org, latent

    ## cls token corrupt
    # def forward(self, imgs, mask_ratio=0.75, return_attention=False):
    #     if return_attention:
    #         latent, mask, ids_restore, embeddings, attention_maps, query, key, value = self.forward_encoder(imgs, mask_ratio, return_attention=True)
    #     else:
    #         latent, mask, ids_restore, embeddings = self.forward_encoder(imgs, mask_ratio)

    #     pred_clean = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    #     loss_clean = self.forward_loss(imgs, pred_clean, mask)
    #     sigmas = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    #     # Decode for each sigma after corrupting ONLY the CLS token
    #     preds_by_sigma = {}
    #     latent_corrupted = {}
    #     for sigma in sigmas:
    #         if float(sigma) == 0.0:
    #             preds_by_sigma[0.0] = pred_clean
    #             continue
    #         noise = torch.randn_like(latent[:, 0, :]) * float(sigma)
    #         latent_noisy = latent.clone()
    #         latent_noisy[:, 0, :] = latent_noisy[:, 0, :] + noise
    #         latent_corrupted[float(sigma)] = latent_noisy
    #         preds_by_sigma[float(sigma)] = self.forward_decoder(latent_noisy, ids_restore)

    #     if return_attention:
    #         return loss_clean, latent_corrupted, preds_by_sigma, mask, latent, embeddings, attention_maps, query, key, value
    #     return loss_clean, latent_corrupted, preds_by_sigma, mask, latent, embeddings

        
    
    # def forward(self, imgs, mask_ratio=0.75):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred1 = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    #     print(pred1.shape)
    #     loss = self.forward_loss(imgs, pred1, mask)
        
    #     latent_corrupt = latent
    #     class_token = latent_corrupt[:, 0, :]  # extract class token
    #     print("Class token shape:", class_token.shape)
    #     # print(class_token)
    #     # corrupt class token then give to decoder
    #     corrupted_class_token = class_token + torch.randn_like(class_token) * 3  # Add noise for corruption
    #     # print("Corrupted class token shape:", corrupted_class_token.shape)
    #     # print(corrupted_class_token)
    #     # Replace the original class token with the corrupted version
    #     latent_corrupt[:, 0, :] = corrupted_class_token

    #     pred2 = self.forward_decoder(latent_corrupt, ids_restore)  # [N, L, p*p*3]
    #     # loss = self.forward_loss(imgs, pred2, mask)
    #     return loss, pred1, pred2, mask, latent

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

