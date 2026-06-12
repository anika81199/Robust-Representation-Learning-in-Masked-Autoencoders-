""" 
Purpose:
    Reproduce the subspace-based analysis for MAE

Paper Section:
    Section 4.1 Class-wise subspace geometry across network depth

Reproduces:
    Figure 3 (a) Layer-wise distribution of principal angles between classes across layers
    Figure 3 (b) Layer-wise evolution of the minimum singular value across classes

Usage:
    python analysis/subspace_geometry.py

Output:
    final/theta1_boxplot.png
    final/minSVplot.png
"""

import os
from sklearn.utils import shuffle
from typing import List, Dict, Any, Optional
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import models_mae                                      

np.float = float

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(img, title=''):
    # image is [H, W, 3]
    assert img.shape[2] == 3
    image = torch.tensor(img)
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

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

data_dir_class1 = "ImageNet/n02106662"     # 235, german shepherd
data_dir_class2 = "ImageNet/n01855032"     # 98,  red-breasted
data_dir_class3 = "Imagenet/n02107312"     # 237, miniature pinscher
data_dir_class4 = "ImageNet/n02105641"     # 229, old sheep dog
data_dir_class5 = "ImageNet/n02782093"     # 417, balloon
data_dir_class6 = "ImageNet/n02788148"     # 421, bannister
data_dir_class7 = "ImageNet/n02802426"     # 430, basketball
data_dir_class8 = "ImageNet/n03788195"     # 668, mosque
data_dir_class9 = "ImageNet/n04065272"     # 757, RV
data_dir_class10 = "ImageNet/n04273569"    # 814, speedboat

image_paths_class1 = [os.path.join(data_dir_class1, img) for img in os.listdir(data_dir_class1) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class2 = [os.path.join(data_dir_class2, img) for img in os.listdir(data_dir_class2) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class3 = [os.path.join(data_dir_class3, img) for img in os.listdir(data_dir_class3) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class4 = [os.path.join(data_dir_class4, img) for img in os.listdir(data_dir_class4) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class5 = [os.path.join(data_dir_class5, img) for img in os.listdir(data_dir_class5) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class6 = [os.path.join(data_dir_class6, img) for img in os.listdir(data_dir_class6) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class7 = [os.path.join(data_dir_class7, img) for img in os.listdir(data_dir_class7) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class8 = [os.path.join(data_dir_class8, img) for img in os.listdir(data_dir_class8) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class9 = [os.path.join(data_dir_class9, img) for img in os.listdir(data_dir_class9) if img.endswith(('.png', '.JPEG', '.jpeg'))]
image_paths_class10 = [os.path.join(data_dir_class10, img) for img in os.listdir(data_dir_class10) if img.endswith(('.png', '.JPEG', '.jpeg'))]

processed_images_class1 = preprocess_images(image_paths_class1)   # 50 images in each class
processed_images_class2 = preprocess_images(image_paths_class2)    
processed_images_class3 = preprocess_images(image_paths_class3)    
processed_images_class4 = preprocess_images(image_paths_class4)    
processed_images_class5 = preprocess_images(image_paths_class5)    
processed_images_class6 = preprocess_images(image_paths_class6)    
processed_images_class7 = preprocess_images(image_paths_class7)    
processed_images_class8 = preprocess_images(image_paths_class8)    
processed_images_class9 = preprocess_images(image_paths_class9)    
processed_images_class10 = preprocess_images(image_paths_class10)    

processed_images_by_class = [processed_images_class1, processed_images_class2, processed_images_class3, processed_images_class4, 
                             processed_images_class5, processed_images_class6, processed_images_class7, 
                             processed_images_class8, processed_images_class9, processed_images_class10]

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def extract_latent(images, model, mask_ratio = 0.75):
    latent_tensors = []     # class tokens --> (1,768)
    matrices=[]             # latent shape --> (50, 768)    
    reduced = []            # patch tokens --> (49, 768)
    positions = []          # of visible patches --> (B, 49)
    layerwise_embeddings = []  # Store embeddings from each layer   (12, N, 768)  N: (no. of patches + 1)
    layerwise_attn_weights = []  # (12,12,N,N)
    layerwise_query = []
    layerwise_key = []
    layerwise_value = []

    for idx, img in enumerate(images):
        print(f"\nProcessing Image {idx + 1}/{len(images)}")

        x = torch.tensor(img)  
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)  

        # Run MAE model
        with torch.no_grad():
            loss, y, mask, latent, layer_embeddings, attn_maps, query, key, value = model(x.float(), mask_ratio=mask_ratio, return_attention=True)   # change masking ratio and check allignment
            matrix = latent.squeeze(0)               # (1x50x768)
            print("Matrix shape:", matrix.shape)     # (50x768)
            matrix_numpy = matrix.detach().numpy()

            reduced_matrix = matrix[1:, :]         # (49, 768)
            print("Reduced matrix shape:", reduced_matrix.shape)
            reduced_matrix = reduced_matrix.detach().numpy()

            # Retain only the first token (class token)
            class_token = latent[:, 0, :]  # Shape: [1, 1, 768]
            print("Class token shape:", class_token.shape) 

            # Extract visible patch position (1 is removing, 0 is keeping)
            mask_indices = np.where(mask.squeeze(0).detach().cpu().numpy() == 0)[0]  
            # print("Mask indices (visible patches):", mask_indices) 

            # Store layerwise embeddings
            layer_embeddings_np = [layer.squeeze(0).detach().cpu().numpy() for layer in layer_embeddings]
            layerwise_embeddings.append(layer_embeddings_np)

            # Store layer-wise attention maps
            attn_maps_np = [attn.detach().cpu().numpy() for attn in attn_maps]
            layerwise_attn_weights.append(attn_maps_np)

            # Store layer-wise q, k, v
            query_np = [q.detach().cpu().numpy() for q in query]
            layerwise_query.append(query_np)
            key_np = [k.detach().cpu().numpy() for k in key]
            layerwise_key.append(key_np)
            value_np = [v.detach().cpu().numpy() for v in value]
            layerwise_value.append(value_np)
            
        latent_tensors.append(class_token.squeeze(0))  #  [1, 768]
        matrices.append(matrix_numpy)         # 50x768 stored for each images
        reduced.append(reduced_matrix)
        positions.append(mask_indices)  # Indices of visible patches

    latent_tensors = torch.stack(latent_tensors)  # Final shape: [N, 768]
    return latent_tensors, matrices, reduced, positions, layerwise_embeddings, layerwise_attn_weights, layerwise_query, layerwise_key, layerwise_value   # tensor, list

chkpt_dir = 'mae_pretrain_vit_base.pth'           
model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
print('Model loaded.')

def layer_subspace_dims_svd(H, include_cls: bool = False):

    H = torch.tensor(H)
    assert H.dim() == 4, "H must be [B, L, N, D]"
    B, L, N, D = H.shape
    S_list, V_list = [], []

    for li in range(L):
        X = H[:, li, :, :]
        if not include_cls:
            X = X[:, 1:, :]          # drop CLS → [B, N-1, D]
        X = X.reshape(-1, D).float() # [M, D]
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu                  # [M, D]
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)        # Vh is Vt, so orthonormal cols becomes rows
        # print(Vh.shape)
        # print(f"Layer {li:02d} - Singular values:")
        # print(S)
        S_list.append(S.cpu())  
        V_list.append(Vh.cpu())

    return S_list, V_list

def get_topk_basis_from_Vh(Vh: torch.Tensor, k: int):
    V_top = Vh[:k, :].T  # top-k rows then transpose
    return V_top  # [D, k]

COLOR_CYCLE = ["lightcoral", "navy", "green", "purple", "orange", "gray",
               "pink", "brown", "olive", "cyan", "steelblue", "plum"]

TOPK = 10
INCLUDE_CLS = False
CLASS_NAMES = [f"Class {i}" for i in range(10)]

all_S_by_class = []     # list of length 10; each item: list length L (torch tensors of SV)
all_Vh_by_class = []    # same for Vh

for ci, images in enumerate(processed_images_by_class):
    _, _, _, _, layerwise_embeddings, attn_maps, _, _, _ = extract_latent(images, model_mae)
    layerwise_embeddings = np.array(layerwise_embeddings)  # [B, L, N, D]
    S_list, Vh_list = layer_subspace_dims_svd(layerwise_embeddings, include_cls=INCLUDE_CLS)
    all_S_by_class.append(S_list)
    all_Vh_by_class.append(Vh_list)

plt.figure(figsize=(10, 6))
for ci in range(10):
    S_list = all_S_by_class[ci]
    num_layers = len(S_list)
    layers = np.arange(1, num_layers + 1)
    mins = [S.min().item() for S in S_list]
    plt.plot(
        layers,
        mins,
        marker='o',
        label=CLASS_NAMES[ci])
plt.xlabel("Layer", fontsize=14)
plt.ylabel("Minimum singular value", fontsize=14)
plt.xticks(np.arange(1, num_layers + 1), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.legend(fontsize=11)
plt.tight_layout()
# plt.savefig("final/minSVplot.png", dpi=300)
plt.show()


# PRINCIPAL ANGLES BETWEEN CLASSES
K = 10
num_classes = 10
CLASS_NAMES = [f"Class {i}" for i in range(num_classes)]  

bases_by_layer = []
num_layers = len(all_Vh_by_class[0])  

for li in range(num_layers):
    layer_bases = []
    for ci in range(num_classes):
        Vh = all_Vh_by_class[ci][li]          # torch [D, D]
        V  = get_topk_basis_from_Vh(Vh, K)    # [D, K]
        layer_bases.append(V)                  # for a fixed layer l and class c, Vk of every class
    bases_by_layer.append(layer_bases)         # (12 layers × 10 classes × D x K)

def principal_angles(V1, V2):
    
    M = V1.T @ V2                   # K x K
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, -1.0, 1.0) 
    return torch.arccos(S) , S # angles in radians

def pairwise_principal_angles(layer_bases, class_names=None):
    """
    layer_bases: list of [D,K] torch.Tensors, one per class
    Returns:
        dict[(i, j)] = np.ndarray of principal angles (in degrees)
    Computes principal angles for all class pairs (i, j),
    including same-class (i == j) pairs.
    """
    results = {}
    num_classes = len(layer_bases)
    for i in range(num_classes):
        for j in range(num_classes):
            ang, cos = principal_angles(layer_bases[i], layer_bases[j])
            ang_deg = torch.rad2deg(ang).cpu().numpy()
            # print(f"Class {i} vs Class {j} → cos={np.round(cos.numpy(), 3)}")
            results[(i, j)] = ang_deg
    return results

angles_by_layer = []

for li in range(num_layers):
    layer_bases = bases_by_layer[li]        # 10, 768, 10 -> class x D x K
    # print(f"\n=== Layer {li} ===")
    results = pairwise_principal_angles(layer_bases, CLASS_NAMES)
    angles_by_layer.append(results)
    
def summarize_angle_dict(angle_dict, mode="min"):
    num_classes = max(max(i,j) for (i,j) in angle_dict.keys()) + 1
    mat = np.zeros((num_classes, num_classes))
    for (i,j), angs in angle_dict.items():
        if mode == "min":
            val = np.min(angs)
        elif mode == "mean":
            val = np.mean(angs)
        elif mode == "max":
            val = np.max(angs)
        mat[i,j] = mat[j,i] = val
    return mat
mats = [summarize_angle_dict(angle_dict, mode="min") for angle_dict in angles_by_layer]

# Collect θ1 distributions per layer
theta1_by_layer = []   # list of length L; each entry is a list of θ1 values
for li, angle_dict in enumerate(angles_by_layer):
    theta1_vals = []
    for (i, j), angs in angle_dict.items():
        if i < j:                    # avoid duplicates and diagonal
            theta1_vals.append(np.min(angs))   # θ1
    theta1_by_layer.append(theta1_vals)
for li, vals in enumerate(theta1_by_layer):
    print(f"Layer {li+1}: {len(vals)} θ1 values, range [{min(vals):.2f}, {max(vals):.2f}]")

# box-plot
plt.figure(figsize=(12, 4))
sns.boxplot(data=theta1_by_layer, showfliers=False, color="white",boxprops=dict(edgecolor='black', linewidth=1.5),
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2))
plt.xlabel("Encoder Layer", fontsize=12)
plt.ylabel(r"Pairwise $\theta_1$ (in degrees)", fontsize=12)
plt.xticks(
    ticks=range(len(theta1_by_layer)),
    labels=[f"L{l+1}" for l in range(len(theta1_by_layer))])
plt.tight_layout()
# plt.savefig("final/theta1_boxplot.png", dpi=300)
plt.show()
