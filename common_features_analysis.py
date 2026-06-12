""" 
Purpose:
    Reproduce the common feature analysis results for attention-guided occluded and blurred inputs

Paper Section:
    Section 4.3 Robustness of latent representations

Reproduces:
    Figure 5 Average drop from C(l,h) clean for 
             (a) Gaussian blur and (b) attention-guided occlusion perturbations.

Usage:
    python analysis/common_features_analysis.py

Output:
    Heatmap for feature retention for occluded and blurred images
    Average drops plot for occluded and blurred images
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import models_mae 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

np.float = float
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

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

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

chkpt_dir= 'mae_finetuned_vit_base.pth'           
model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
print('Model loaded.')

def run_one_image(img, model, mask_ratio=0.75):    
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

data_dir_class1 = "ImageNet/n02106662"   # (dogs)
image_paths_class1 = [os.path.join(data_dir_class1, img) for img in os.listdir(data_dir_class1) if img.endswith(('.png', '.JPEG', '.jpeg'))]
processed_images_class1 = preprocess_images(image_paths_class1)

# ================================================================================
# --------------------------------- A.V analysis (occluded) ----------------------
# ================================================================================

occlusion_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

def load_occluded_images(level):
    level_dir = f"occluded_{level}"
    class_to_paths = {}

    if not os.path.exists(level_dir):
        return None

    for sub in sorted(os.listdir(level_dir)):
        subfolder = os.path.join(level_dir, sub)
        if not os.path.isdir(subfolder):
            continue

        imgs = [
            os.path.join(subfolder, fn)
            for fn in sorted(os.listdir(subfolder))
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        class_to_paths[sub] = imgs  

    return class_to_paths

def preprocess_images(image_paths):
    processed_images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))  
        image = np.array(image) / 255.0
        image = (image - imagenet_mean) / imagenet_std  # Apply standardization
        processed_images.append(image)
    return processed_images

#  function to extract top-k common features for each class
def get_common_features_for_class(images, class_id, top_k=10, threshold=6, num_layers=12, num_heads=12):
    from collections import defaultdict

    image_ids = [f"img_{i+1:03}" for i in range(len(images))]
    results = []

    for img, img_id in zip(images, image_ids):
        print(f"[Class {class_id}] Processing {img_id}...")
        attn_map, attn_mapss, mask_indices, queries, keys, values, layer_embeddings_np, masked_img = run_one_image(img, model_mae, mask_ratio=0.0)

        for layer in range(num_layers):
            for head in range(num_heads):
                A = attn_mapss[layer, head]
                V = values[layer, head]
                O = A @ V

                g_patch = O[1:].mean(axis=0)
                g_cls = O[0]

                g_patch = torch.tensor(g_patch) if not isinstance(g_patch, torch.Tensor) else g_patch
                g_cls = torch.tensor(g_cls) if not isinstance(g_cls, torch.Tensor) else g_cls

                topk_patch = torch.topk(g_patch, k=top_k).indices.cpu().numpy().tolist()
                topk_cls = torch.topk(g_cls, k=top_k).indices.cpu().numpy().tolist()

                results.append({
                    "image_id": img_id,
                    "layer": layer,
                    "head": head,
                    "topk_patch_indices": topk_patch,
                    "topk_cls_indices": topk_cls
                })

    df = pd.DataFrame(results)
    grouped = df.groupby(["layer", "head"])
    final_results = []

    for (layer, head), group in grouped:
        patch_features = [feat for sublist in group["topk_patch_indices"] for feat in sublist]
        cls_features = [feat for sublist in group["topk_cls_indices"] for feat in sublist]

        patch_counter = Counter(patch_features)
        cls_counter = Counter(cls_features)

        common_patch = [feat for feat, count in patch_counter.items() if count >= threshold]
        common_cls = [feat for feat, count in cls_counter.items() if count >= threshold]

        final_results.append({
            "layer": layer,
            "head": head,
            f"patch_common_features_cls{class_id}": sorted(common_patch),
            f"patch_count_cls{class_id}": len(common_patch),
            f"cls_common_features_cls{class_id}": sorted(common_cls),
            f"cls_count_cls{class_id}": len(common_cls)
        })

    return pd.DataFrame(final_results)

all_occ_dfs = []

# for level in occlusion_levels:
#     print(f"\n========== Processing Occlusion Level {level}% ==========")

#     occluded = load_occluded_images(level)
#     if occluded is None:
#         print(f"⚠️ No folder occluded_{level}/, skipping…")
#         continue

#     if "n02106662" not in occluded:
#         print(f"⚠️ n02106662 not found in occluded_{level}, skipping…")
#         continue

#     class1_paths = occluded["n02106662"]

#     # Preprocess images into tensors
#     occluded_imgs_cls1 = preprocess_images(class1_paths)

#     occ_df_cls1 = get_common_features_for_class(
#         images=occluded_imgs_cls1[:10],
#         class_id=1,
#         top_k=10,
#         threshold=6,
#         num_layers=12,
#         num_heads=12,
#     )
#     suffix = f"occ{level}"

#     occ_df_cls1 = occ_df_cls1.rename(columns={
#         "patch_common_features_cls1": f"patch_common_features_{suffix}_cls1",
#         "patch_count_cls1":          f"patch_count_{suffix}_cls1",
#         "cls_common_features_cls1":  f"cls_common_features_{suffix}_cls1",
#         "cls_count_cls1":            f"cls_count_{suffix}_cls1",
#     })

#     all_occ_dfs.append(occ_df_cls1)

# final_occ_df = reduce(
#     lambda left, right: pd.merge(left, right, on=["layer", "head"], how="outer"),
#     all_occ_dfs
# )
# final_occ_df = final_occ_df.sort_values(by=["layer", "head"]).reset_index(drop=True)
# final_occ_df.to_csv("final/occs_0mask.csv", index=False)
# print("Saved final/occs_0mask.csv")
# print(final_occ_df.head())

# --------------------------------------------------------------------------
# VISUALIZATION, occs_0mask.csv contains clean + occluded retained features
# --------------------------------------------------------------------------
df = pd.read_csv("final/occs_0mask.csv", header=1)   
print(df.head())

occ_settings = [
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90
]

patch_cols = [c for c in df.columns 
              if "patch_count" in c and "_cls1" not in c]
cls_cols   = [c for c in df.columns if "cls_count" in c and "_cls1" not in c]
print("PATCH:", patch_cols)
print("CLS:", cls_cols)

occ_labels = [f"{occ}" for occ in occ_settings]

fig, axes = plt.subplots(4, 3, figsize=(26, 16))
axes = axes.flatten()
for idx in range(12):
    ax = axes[idx]
    layer_index = idx        # true index
    layer_label = idx + 1    # human readable
    layer_df = df[df["layer"] == layer_index][["head"] + patch_cols]
    full_mat = pd.DataFrame({"head": np.arange(12)})
    layer_df = full_mat.merge(layer_df, on="head", how="left")
    data_mat = layer_df[patch_cols].fillna(0)
    sns.heatmap(
        data_mat,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=.3,
        cbar=False,
        annot_kws={"size": 14}
    )
    ax.set_title(f"Layer {layer_label}", fontsize=10)
    ax.set_ylabel("Head", fontsize=8)
    ax.set_xlabel("Occlusion (%)", fontsize=8)
    ax.set_xticks(np.arange(len(occ_labels)) + 0.5)
    ax.set_xticklabels(occ_labels, rotation=45, ha="right")
plt.tight_layout()
plt.savefig("final/occs_cls1_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

clean_vals = df["patch_count_cls1"]  # clean patch count
drops = []
for col in patch_cols:
    drops.append((clean_vals - df[col]).mean())
# print(drops)
plt.figure(figsize=(10,5))
plt.plot(occ_labels, drops, marker='o')
plt.xticks(rotation=45)
plt.xlabel("Occlusion (%)")
plt.ylabel("Average drop")
plt.title("Drop in Patch Token Retention vs Occlusion level")
plt.grid(False)
plt.tight_layout()
# plt.savefig("final/occs_drops.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================================================================
# --------------------------------- A.V analysis (blurs) -------------------------
# ================================================================================

blur_settings = [
    (5, 1.0),
    (5, 2.0),
    (5, 4.0),
    (5, 9.0),
    (7, 2.0),
    (7, 4.0),
    (7, 13.5),
    (7, 15.0),
    (11, 2.0),
    (11, 5.0)
]

def apply_gaussian_blur_cv2(image_array, ksize, sigma):
    return cv2.GaussianBlur(image_array, (ksize, ksize), sigma)

def preprocess_images_blurred(image_paths, ksize, sigma):
    processed_images_blur = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB").resize((224, 224))
        image_np = np.array(image) / 255.0
        image_blurred = apply_gaussian_blur_cv2(image_np, ksize, sigma)
        image_blurred = (image_blurred - imagenet_mean) / imagenet_std
        processed_images_blur.append(image_blurred)
    return processed_images_blur

#  function to extract top-k common features for each class
def get_common_features_for_class(images, class_id, top_k=10, threshold=6, num_layers=12, num_heads=12):
    from collections import defaultdict

    image_ids = [f"img_{i+1:03}" for i in range(len(images))]
    results = []

    for img, img_id in zip(images, image_ids):
        print(f"[Class {class_id}] Processing {img_id}...")
        attn_map, attn_mapss, mask_indices, queries, keys, values, layer_embeddings_np, masked_img = run_one_image(img, model_mae, mask_ratio=0.0)

        for layer in range(num_layers):
            for head in range(num_heads):
                A = attn_mapss[layer, head]
                V = values[layer, head]
                O = A @ V

                g_patch = O[1:].mean(axis=0)
                g_cls = O[0]

                g_patch = torch.tensor(g_patch) if not isinstance(g_patch, torch.Tensor) else g_patch
                g_cls = torch.tensor(g_cls) if not isinstance(g_cls, torch.Tensor) else g_cls

                topk_patch = torch.topk(g_patch, k=top_k).indices.cpu().numpy().tolist()
                topk_cls = torch.topk(g_cls, k=top_k).indices.cpu().numpy().tolist()

                results.append({
                    "image_id": img_id,
                    "layer": layer,
                    "head": head,
                    "topk_patch_indices": topk_patch,
                    "topk_cls_indices": topk_cls
                })

    df = pd.DataFrame(results)
    grouped = df.groupby(["layer", "head"])
    final_results = []

    for (layer, head), group in grouped:
        patch_features = [feat for sublist in group["topk_patch_indices"] for feat in sublist]
        cls_features = [feat for sublist in group["topk_cls_indices"] for feat in sublist]

        patch_counter = Counter(patch_features)
        cls_counter = Counter(cls_features)

        common_patch = [feat for feat, count in patch_counter.items() if count >= threshold]
        common_cls = [feat for feat, count in cls_counter.items() if count >= threshold]

        final_results.append({
            "layer": layer,
            "head": head,
            f"patch_common_features_cls{class_id}": sorted(common_patch),
            f"patch_count_cls{class_id}": len(common_patch),
            f"cls_common_features_cls{class_id}": sorted(common_cls),
            f"cls_count_cls{class_id}": len(common_cls)
        })

    return pd.DataFrame(final_results)

all_blur_dfs = []

# for (ksize, sigma) in blur_settings:
#     print(f"\n=== Processing Class 1 with Blur (ksize={ksize}, sigma={sigma}) ===")

#     blurred_images_cls1 = preprocess_images_blurred(image_paths_class1, ksize, sigma)

#     blur_df_cls1 = get_common_features_for_class(
#         images=blurred_images_cls1[:10],   # or just blurred_images_cls1 if you want all
#         class_id=1,
#         top_k=10,
#         threshold=6,
#         num_layers=12,
#         num_heads=12,
#     )

#     blur_suffix = f"k{ksize}_s{sigma}"  # e.g. "k5_s1.0"

#     blur_df_cls1 = blur_df_cls1.rename(columns={
#         f"patch_common_features_cls1": f"patch_common_features_{blur_suffix}_cls1",
#         f"patch_count_cls1":          f"patch_count_{blur_suffix}_cls1",
#         f"cls_common_features_cls1":  f"cls_common_features_{blur_suffix}_cls1",
#         f"cls_count_cls1":            f"cls_count_{blur_suffix}_cls1",
#     })

#     # 4) Store this per-blur dataframe for later merging
#     all_blur_dfs.append(blur_df_cls1)

# # 5) Merge all blur-case DataFrames horizontally on (layer, head)
# final_df = reduce(
#     lambda left, right: pd.merge(left, right, on=["layer", "head"], how="outer"),
#     all_blur_dfs
# )

# final_df = final_df.sort_values(by=["layer", "head"]).reset_index(drop=True)
# final_df.to_csv("final/blurs_class1_0mask.csv", index=False)
# print("Saved: final/blurs_class1_0mask.csv")
# print(final_df.head())

# ------------------------------------------------------------------------
# VISUALIZATION, blurs_class1_0mask.csv contains clean + blur retained features
# ------------------------------------------------------------------------

df = pd.read_csv("final/blurs_class1_0mask.csv", header=1)   
print(df.head())

blur_settings = [
    (5,1.0), (5,2.0), (5,4.0), (5,9.0),
    (7,2.0), (7,4.0), (7,13.5), (7,15.0),
    (11,2.0), (11,5.0)
]

patch_cols_all = [c for c in df.columns if "patch_count" in c]

# clean column
clean_patch_col = [c for c in patch_cols_all if "_cls1" in c][0]

# blur columns
blur_patch_cols = [c for c in patch_cols_all if "_cls1" not in c]

# final ordering for heatmap
patch_cols = [clean_patch_col] + blur_patch_cols

n_blurs = len(blur_settings)
roman_levels = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
blur_labels = roman_levels[:n_blurs]

# Prepend clean reference
full_labels = ["Clean"] + blur_labels

# ---- 3x4 GRID (already correct) ----
fig, axes = plt.subplots(4, 3, figsize=(26, 16))
axes = axes.flatten()

for idx in range(12):
    ax = axes[idx]
    layer_index = idx
    layer_label = idx + 1

    layer_df = df[df["layer"] == layer_index][["head"] + patch_cols]

    full_mat = pd.DataFrame({"head": np.arange(12)})
    layer_df = full_mat.merge(layer_df, on="head", how="left")
    data_mat = layer_df[patch_cols].fillna(0)

    sns.heatmap(
        data_mat,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=.3,
        cbar=False,
        annot_kws={"size": 14}  
    )

    ax.set_title(f"Layer {layer_label}", fontsize=10)
    ax.set_ylabel("Head", fontsize=8)
    ax.set_xlabel("Blur level", fontsize=8)
    ax.set_xticks(np.arange(len(full_labels)) + 0.5)
    ax.set_xticklabels(full_labels, rotation=90, ha="right")

plt.tight_layout()
# plt.savefig("final/blurs_cls1_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# Drop plot
clean_vals = df["patch_count_cls1"]  # clean patch count
drops = []
for col in blur_patch_cols:
    drops.append((clean_vals - df[col]).mean())
plt.figure(figsize=(10,5))
plt.plot(blur_labels, drops, marker='o')
plt.xticks(rotation=45)
plt.xlabel("Blur level")
plt.ylabel("Average drop")
plt.title("Drop in Patch Token Retention vs Blur")
plt.grid(False)
plt.tight_layout()
# plt.savefig("final/blurs_drops.png", dpi=300, bbox_inches='tight')
plt.show()










