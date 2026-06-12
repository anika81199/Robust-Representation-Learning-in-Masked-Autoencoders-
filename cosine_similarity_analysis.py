""" 
Purpose:
    Reproduce the cosine similarity results for attention-guided occluded and blurred inputs

Paper Section:
    Section 4.3 Robustness of latent representations

Reproduces:
    Table 2 Mean cosine similarity between clean and perturbed embeddings for 
            (a) Gaussian blur and (b) attention-guided occlusion

Usage:
    python analysis/cosine_similarity_analysis.py

Output:
    cosine similarity csv
    mean cosine similarity for each perturbation level
"""

import torch.nn as nn
import os
import torch
import numpy as np
import cv2
import pandas as pd
import csv
from PIL import Image
import models_mae 
from sklearn.metrics.pairwise import cosine_similarity


np.float = float
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# ================================================================================
# ------------------------ OCCLUSION vs CLEAN (cosine sim) -----------------------
# ================================================================================

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def load_clean_images(classes, limit=None):
    class_to_paths = {}
    for cls_id, folder in classes.items():
        imgs = [
            os.path.join(folder, fn)
            for fn in sorted(os.listdir(folder))
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if limit:
            imgs = imgs[:limit]
        class_to_paths[cls_id] = imgs
    return class_to_paths

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
        class_to_paths[sub] = imgs  # sub = basename like 'n02106662'

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

def extract_latent(images, model, mask_ratio = 0.0):
    latent_tensors = []     # class tokens --> (1,768)
    matrices=[]             # latent shape --> (50, 768)    
    reduced = []            # patch tokens --> (49, 768)
    positions = []          # of visible patches --> (10, 49)
    layerwise_embeddings = []  # Store embeddings from each layer

    for idx, img in enumerate(images):
        # print(f"\nProcessing Image {idx + 1}/{len(images)}")

        x = torch.tensor(img)  
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)  

        # Run MAE model
        with torch.no_grad():
            loss, y, mask, latent, layer_embeddings = model(x.float(), mask_ratio=mask_ratio)   
            matrix = latent.squeeze(0)               # (1x50x768)
            # print("Matrix shape:", matrix.shape)     # (50x768)
            matrix_numpy = matrix.detach().numpy()

            reduced_matrix = matrix[1:, :]         # (49, 768)
            # print("Reduced matrix shape:", reduced_matrix.shape)
            reduced_matrix = reduced_matrix.detach().numpy()

            # Retain only the first token (class token)
            class_token = latent[:, 0, :]  # Shape: [1, 1, 768]
            # print("Class token shape:", class_token.shape) 

            # Extract visible patch position (1 is removing, 0 is keeping)
            mask_indices = np.where(mask.squeeze(0).detach().cpu().numpy() == 0)[0]  
            # print("Mask indices (visible patches):", mask_indices) 

            # Store layerwise embeddings
            layer_embeddings_np = [layer.squeeze(0).detach().cpu().numpy() for layer in layer_embeddings]
            layerwise_embeddings.append(layer_embeddings_np)
            
        latent_tensors.append(class_token.squeeze(0))  #  [1, 768]
        matrices.append(matrix_numpy)         # 50x768 stored for each images
        reduced.append(reduced_matrix)
        positions.append(mask_indices)  # Indices of visible patches

    latent_tensors = torch.stack(latent_tensors)  # Final shape: [N, 768]
    return latent_tensors, matrices, reduced, positions, layerwise_embeddings     # tensor, list

chkpt_dir = 'mae_finetuned_vit_base.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
print('Model loaded.')

def compute_clean_embeddings(model_mae, classes, limit_per_class=50):
    """
    Returns:
      clean_embeddings[cls_id] = array of shape (N, 768)
    """
    clean_paths_dict = load_clean_images(classes, limit=limit_per_class)
    clean_embeddings = {}
    clean_magnitudes = {}

    for cls_id, clean_paths in clean_paths_dict.items():
        print(f"Extracting clean embeddings for class {cls_id} ...")
        clean_imgs = preprocess_images(clean_paths)
        _, _, _, _, clean_layerwise = extract_latent(clean_imgs, model_mae)
        clean_layerwise = np.array(clean_layerwise)     # (N, 12, 197, 768)
        final_clean = clean_layerwise[:, -1, :, :]      # (N, 197, 768)
        patch_clean = final_clean[:, 1:, :]             # (N, 196, 768)
        clean_emb = np.mean(patch_clean, axis=1)        # (N, 768)
        clean_embeddings[cls_id] = clean_emb
        clean_magnitudes[cls_id] = np.linalg.norm(clean_emb, axis=1)
    return clean_embeddings, clean_magnitudes

def compute_cosine_for_occlusion_level(model_mae, classes, level,
                                       clean_embeddings, clean_magnitudes):
    rows = []
    occ_paths_dict = load_occluded_images(level)
    if occ_paths_dict is None:
        print(f"occluded_{level} not found.")
        return []
    for cls_id, clean_emb in clean_embeddings.items():
        class_basename = os.path.basename(classes[cls_id])
        occ_paths = occ_paths_dict[class_basename]
        occ_imgs = preprocess_images(occ_paths)
        _, _, _, _, occ_layerwise = extract_latent(occ_imgs, model_mae)
        occ_layerwise = np.array(occ_layerwise)
        final_occ = occ_layerwise[:, -1, :, :]      # (N, 50, 768)
        patch_occ = final_occ[:, 1:, :]             # (N, 49, 768)
        occ_emb = np.mean(patch_occ, axis=1)        # (N, 768)
        occ_magnitudes = np.linalg.norm(occ_emb, axis=1)
        clean_mag = clean_magnitudes[cls_id]
        n = min(len(clean_emb), len(occ_emb))
        for i in range(n):
            cos_sim = float(
                cosine_similarity(clean_emb[i][None], occ_emb[i][None])[0][0])
            rows.append({
                "class_id": cls_id,
                "img_idx": i,
                "occlusion_level": level,
                "cos_sim": cos_sim,
                "clean_mag": clean_mag[i],
                "occ_mag": occ_magnitudes[i],
                "mag_diff": abs(clean_mag[i] - occ_magnitudes[i])
            })
    return rows

occlusion_levels = [0,10,20,30,40,50,60,70,80,90]
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

clean_embeddings, clean_magnitudes = compute_clean_embeddings(
    model_mae, classes, limit_per_class=50
)
all_rows = []
for level in occlusion_levels:
    print(f"\n=== Running occlusion level {level}% ===")
    rows = compute_cosine_for_occlusion_level(
        model_mae, classes, level,
        clean_embeddings, clean_magnitudes
    )
    all_rows.extend(rows)

df_occ = pd.DataFrame(all_rows)
df_occ.to_csv("final/cosine_similarity_finetune_occlusion.csv", index=False)
print("Saved → final/cosine_similarity_finetune_occlusion.csv")
df_occ = pd.read_csv("final/cosine_similarity_finetune_occlusion.csv")
summary = df_occ.groupby("occlusion_level")["cos_sim"].mean().reset_index()
print(summary)

# def print_magnitudes_for_image(df, cls_id, img_idx):
#     df_img = df[
#         (df["class_id"] == cls_id) &
#         (df["img_idx"] == img_idx)
#     ].sort_values("occlusion_level")

#     clean_mag = df_img["clean_mag"].iloc[0]

#     print(f"\nClass {cls_id}, Image {img_idx}")
#     print(f"Clean magnitude: {clean_mag:.3f}")
#     print("Occlusion level → occ_mag (Δ from clean)")

#     for _, row in df_img.iterrows():
#         print(
#             f"{int(row['occlusion_level']):>3}% → "
#             f"{row['occ_mag']:.3f} "
#             f"(Δ {row['occ_mag'] - clean_mag:+.3f})"
#         )
# print_magnitudes_for_image(df_occ, cls_id=235, img_idx=0)

# ================================================================================
# --------------------------------- BLURRED vs CLEAN ------------------------------
# ================================================================================
def preprocess_images(image_paths):
    processed_images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))  
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - imagenet_mean) / imagenet_std  # Apply standardization
        processed_images.append(image)
    return processed_images

def apply_gaussian_blur_cv2(image_array, ksize, sigma):
    return cv2.GaussianBlur(image_array, (ksize, ksize), sigma)

def preprocess_to_unit_range(image_paths):
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0    # 0–1
        images.append(img)
    return images

def preprocess_image_std(image_np):
    """Normalize H×W×3 np.float32 in [0,1] → standardized [H,W,3]"""
    img = (image_np - imagenet_mean) / imagenet_std
    return img

def load_clean_images(classes, limit=None):
    class_to_paths = {}
    for cls_id, folder in classes.items():
        imgs = [
            os.path.join(folder, fn)
            for fn in sorted(os.listdir(folder))
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if limit:
            imgs = imgs[:limit]
        class_to_paths[cls_id] = imgs
    return class_to_paths

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

def compute_clean_embeddings_mae(model_mae, classes, limit_per_class=50):

    clean_paths_dict = load_clean_images(classes, limit_per_class)
    clean_embeddings = {}      # cls_id → (N, 768)
    clean_paths = {}           # cls_id → list of image paths

    for cls_id, paths in clean_paths_dict.items():
        print(f"\nExtracting CLEAN embeddings for class {cls_id}...")

        imgs = preprocess_images(paths)
        _, _, _, _, layerwise = extract_latent(imgs, model_mae)

        layerwise = np.array(layerwise)             # (N, 12, 197, 768)
        final_layer = layerwise[:, -1, :, :]        # (N, 197, 768)
        patch_tokens = final_layer[:, 1:, :]        # (N, 196, 768)
        clean_emb = patch_tokens.mean(axis=1)       # (N, 768)

        clean_embeddings[cls_id] = clean_emb
        clean_paths[cls_id] = paths

    return clean_embeddings, clean_paths

def compute_cosine_for_blur(model_mae, clean_embeddings, clean_paths, blur_settings):

    rows = []

    for cls_id, paths in clean_paths.items():
        print(f"\nProcessing BLUR for class {cls_id}...")

        clean_emb = clean_embeddings[cls_id]      # (N, 768)
        clean_imgs = preprocess_to_unit_range(paths)
        N = len(clean_imgs)

        for i in range(N):
            img_np = clean_imgs[i]     # raw clean image in 0–1 range
            for (ksize, sigma) in blur_settings:
                print(f"\n blur {ksize},{sigma}...")
                blur_np = apply_gaussian_blur_cv2(img_np, ksize, sigma)
                blur_std = preprocess_image_std(blur_np)
                _, _, _, _, layerwise = extract_latent([blur_std], model_mae)
                layerwise = np.array(layerwise)
                final_layer = layerwise[:, -1, :, :]
                patch_tokens = final_layer[:, 1:, :]
                blur_emb = patch_tokens.mean(axis=1)[0]   # (768,)
                sim = float(cosine_similarity(clean_emb[i][None], blur_emb[None])[0][0])

                rows.append({
                    "class_id": cls_id,
                    "img_idx": i,
                    "ksize": ksize,
                    "sigma": sigma,
                    "cos_sim": sim
                })

    return rows

clean_embeddings, clean_paths = compute_clean_embeddings_mae(
    model_mae,
    classes,
    limit_per_class=50
)
all_rows_blur = []
print("\n===== Running BLUR cosine similarity =====\n")
rows = compute_cosine_for_blur(
    model_mae,
    clean_embeddings,
    clean_paths,
    blur_settings)
all_rows_blur.extend(rows)
df_blur = pd.DataFrame(all_rows_blur)
df_blur.to_csv("final/cosine_similarity_finetune_blur.csv", index=False)
print(df_blur.head())
blur_summary = (
    df_blur.groupby(["ksize", "sigma"])["cos_sim"]
    .mean()
    .reset_index()
    .sort_values(["ksize", "sigma"])
)
print("\n===== BLUR SUMMARY =====")
print(blur_summary)