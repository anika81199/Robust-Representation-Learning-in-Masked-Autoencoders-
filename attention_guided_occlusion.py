""" 
Purpose:
    Reproduce the attention-guided occluded images

Paper Section:
    Used in Sections 4.2 and 4.3

Reproduces:
    Attention-guided occlusion images

Usage:
    python analysis/attention_guided_occlusion.py

Output:
    folders with attention-guided occluded images (10 classes) for 10 different perturbation levels
"""


import sys
import os
import math
import torch
import numpy as np
from PIL import Image
import models_mae                                      
from sklearn.metrics import silhouette_score
from collections import Counter
import torch.nn.functional as F
from scipy.stats import entropy
from collections import defaultdict
import ast
from ast import literal_eval
from functools import reduce
from typing import Dict

np.float = float
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# Set constants
PATCH_SIZE = 16
GRID_SIZE = 14

def patch_id_to_rc(patch_id, grid=GRID_SIZE):
    r = patch_id // grid
    c = patch_id %  grid
    return r, c

def attention_rollout(attn_maps, retain_cls_token=True, head_fusion="mean", discard_ratio=0.0):
    """
    Compute attention rollout from raw attention maps.

    Args:
        attn_maps (torch.Tensor or np.ndarray): Shape (12, 12, 197, 197), raw attention weights.
        retain_cls_token (bool): If True, extracts attention from CLS token, else averages patches.

    Returns:
#         np.ndarray: Attention rollout heatmap, shape (14, 14) for overlaying.
#     """
    if isinstance(attn_maps, np.ndarray):
        attn_maps = torch.tensor(attn_maps)  # Convert to tensor if needed
    
    num_layers, _ , num_tokens, _ = attn_maps.shape  # (12, 12, 197, 197)
    if head_fusion == "mean":
        attn_maps = attn_maps.mean(axis=1)
    elif head_fusion == "max":
        attn_maps = attn_maps.max(axis=1)[0]
    elif head_fusion == "min":
        attn_maps = attn_maps.min(axis=1)[0]
    else:
        raise "Unsupported head_fusion type"

    # print(attn_maps.shape)
    # (12, 197, 197)
    # Drop the lowest attentions, but don't drop the class token
    for layer in range(num_layers):
        attn = attn_maps[layer]
        flat = attn.view(-1)
        num_to_discard = int(flat.shape[0] * discard_ratio)

        # Don't touch CLS token row
        flat_clone = flat.clone()
        cls_start = 0 * num_tokens  # 0th row start
        cls_end = 1 * num_tokens    # 0th row end
        flat_clone[cls_start:cls_end] = float('inf')  # Prevent dropping CLS token weights

        lowest_values, indices = torch.topk(flat_clone, num_to_discard, largest=False)   # flat_clone is only used to find the lowest attention values, while flat is the one you actually zero out
        flat[indices] = 0
        attn_maps[layer] = flat.view(num_tokens, num_tokens)

    # Add identity matrix to account for residual connections
    I = torch.eye(num_tokens).unsqueeze(0).repeat(num_layers, 1, 1)  # (12, 197, 197)
    aug_att_mat = (attn_maps + I)/2                 # equal weights for adding 

    # Normalize so each row sums to 1
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)      # (12, 197, 197)

    # Compute rollout for each layer separately 
    joint_attentions = torch.zeros_like(aug_att_mat)  #(12,197,197)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        row_sums = joint_attentions[n].sum(dim=-1)
        # print(row_sums.shape)
        # print(f"Layer {n}: Row sum min={row_sums.min().item():}, max={row_sums.max().item():}")

    # Extract final attention values for each layer
    attention_rollout_per_layer = []
    grid_size = int(np.sqrt(num_tokens - 1))  # Should be 14

    for i in range(joint_attentions.size(0)):
        v = joint_attentions[i]  # Attention matrix for layer i

        if retain_cls_token:
            mask = v[0, 1:]  # CLS token attending to all 196 patches (196,)
        else:
            mask = v[1:, 1:].mean(dim=0)  # Mean over all patches (196,)
        
        mask = mask / mask.max()  # Normalize
        mask = mask.detach().numpy()
        # print(mask.shape)
        attention_rollout_per_layer.append(mask)

    return attention_rollout_per_layer[-1]  # final attn rollout 196 values

def occlude_top_patches_by_percentile(img, mask, percentile):
    threshold_value = np.percentile(mask, 100 - percentile)
    selected_ids = np.where(mask >= threshold_value)[0]
    # print(f"Selected {len(selected_ids)} patches (above {threshold_value:.4f} threshold)")
    
    occluded = img.copy()
    fill_value = 0.0 if occluded.dtype != np.uint8 else 0

    for pid in selected_ids:
        r, c = patch_id_to_rc(pid)
        r0, r1 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
        c0, c1 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
        occluded[r0:r1, c0:c1, :] = fill_value

    return occluded

# ============= occlusion images create for 10 classes =======================
classes = {
    235: "ImageNet/n02106662",        # 235, german shepherd
    98:  "ImageNet/n01855032",        # 98,  red-breasted
    237: "Imagenet/n02107312",        # 237, miniature pinscher
    229: "ImageNet/n02105641",        # 229, old sheep dog
    417: "ImageNet/n02782093",        # 417, balloon
    421: "ImageNet/n02788148",        # 421, bannister
    430: "ImageNet/n02802426",        # 430, basketball
    668: "ImageNet/n03788195",        # 668, mosque
    757: "ImageNet/n04065272",        # 757, RV
    814: "ImageNet/n04273569",        # 814, speedboat
}

# Preprocess images
def preprocess_images(image_paths):
    processed_images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")  # RGB
        image = image.resize((224, 224))  
        image = np.array(image) / 255.0
        image = (image - imagenet_mean) / imagenet_std  # Apply standardization
        processed_images.append(image)
    return processed_images

def compute_psnr(img1, img2):
    """Compute PSNR between two images in [0, 1] float format"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def inverse_normalize(image, mean, std):
    image = (image * std) + mean  
    # image = np.clip(image * 255, 0, 255).astype(np.uint8)  
    return np.clip(image, 0, 1)


def run_one_image(img, model, mask_ratio=0.0): 
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask, latent, layer_embeddings, attn_maps, queries, keys, values = model(x.float(), mask_ratio=mask_ratio, return_attention=True)
    # latent_tensor = latent.squeeze(0)
    # reduced_matrix = latent_tensor[1:, :].detach().numpy()    # (49, 768)

    layer_embeddings = [layer.squeeze(0).detach().cpu().numpy() for layer in layer_embeddings]
    layer_embeddings_np = np.array(layer_embeddings)
    # print(layer_embeddings_np.shape)

    attn_maps = [attn.detach().cpu().numpy() for attn in attn_maps]
    attn_maps_np = np.array(attn_maps)

    queries = [q.detach().cpu().numpy() for q in queries]
    queries_np = np.array(queries)
    keys = [k.detach().cpu().numpy() for k in keys]
    keys_np = np.array(keys)
    values = [v.detach().cpu().numpy() for v in values]
    values_np = np.array(values)

    # print(attn_maps_np.shape)                  # (12, 1, 12, 197, 197)
    attn_maps_np = attn_maps_np.squeeze(1)  
    # print(attn_maps_np.shape)                # (12, 12, 197, 197)
    queries_np = queries_np.squeeze(1)
    keys_np = keys_np.squeeze(1)                # (12, 12, 197, 64)
    values_np = values_np.squeeze(1)

    # Extract visible patch position (1 is removing, 0 is keeping)
    mask_indices = np.where(mask.squeeze(0).detach().cpu().numpy() == 0)[0] 

    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    mask_binary = mask[0][:, :, 0].numpy()
    visible_images = []
    patch_size = model.patch_embed.patch_size[0]
    h, w = mask_binary.shape
    h_patches = h // patch_size
    w_patches = w // patch_size

    for i in range(h_patches):
        for j in range(w_patches):
            if len(visible_images) == 5:
                break
            patch_mask = mask_binary[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if np.all(patch_mask == 0):  # fully visible patch
                patch = x[0][i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :].numpy()
                visible_images.append(patch)
        if len(visible_images) == 5:
            break

    attn_maps_avg = attn_maps_np.mean(axis=1)  # Shape: (12, 197, 197) (average over heads)
    return attn_maps_avg[:,1:,1:], attn_maps_np, mask_indices, queries_np, keys_np, values_np, layer_embeddings_np, im_masked[0]


occlusion_levels = [0,10,20,30,40,50,60,70,80,90]

# # Loop over occlusion levels
# for level in occlusion_levels:
#     # Create top-level folder for this occlusion level
#     level_dir = f"occluded_{level}"
#     os.makedirs(level_dir, exist_ok=True)
#     total_psnr, total_ssim, total_images = 0.0, 0.0, 0
    
#     for class_path in classes.values():
#         image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.png', '.JPEG', '.jpeg'))]
#         processed_images = preprocess_images(image_paths)
#         class_name = os.path.basename(class_path)  # e.g. "n02106662"
#         class_output_dir = os.path.join(level_dir, class_name)
#         os.makedirs(class_output_dir, exist_ok=True)

#         print(f"\n Processing Class: {class_name} | Occlusion: {level}%")

#         for idx, img in enumerate(processed_images):
#             attn_map, attn_mapss, mask_indices, queries, keys, values, layer_embeddings_np, masked_img = run_one_image(img, model_mae, mask_ratio=0.0)   # masking = 0, fine-tuned model
#             attn_mapss_normalized = normalize_attention(attn_mapss)  # Normalize attention
#             mask = attention_rollout(attn_mapss_normalized, head_fusion="mean", discard_ratio=0.0)
#             occluded_img = occlude_top_patches_by_percentile(img, mask, percentile=level)

#             clean_rgb = inverse_normalize(img, imagenet_mean, imagenet_std)
#             occ_rgb   = inverse_normalize(occluded_img, imagenet_mean, imagenet_std)
#             psnr_val = compute_psnr(clean_rgb, occ_rgb)
#             ssim_val = ssim(clean_rgb, occ_rgb, channel_axis=2, data_range=1.0)

#             total_psnr += psnr_val
#             total_ssim += ssim_val
#             total_images += 1

#             occ_rgb = np.clip(occ_rgb * 255, 0, 255).astype(np.uint8) 
#             # Save image
#             save_path = os.path.join(class_output_dir, f"img_{idx:03d}.png")
#             cv2.imwrite(save_path, cv2.cvtColor(occ_rgb, cv2.COLOR_RGB2BGR))
#             # print(f"Saved: {save_path}")

#     if total_images > 0:
#         mean_psnr = total_psnr / total_images
#         mean_ssim = total_ssim / total_images
#         print(f"\n Occlusion {level}% → Mean PSNR={mean_psnr:.3f} dB | Mean SSIM={mean_ssim:.4f}\n")
#     else:
#         print(f"\n⚠️ No images processed for occlusion {level}%\n")
