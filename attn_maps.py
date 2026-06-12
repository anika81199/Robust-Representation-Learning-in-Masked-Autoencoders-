""" 
Purpose:
    Reproduce the attention distance analysis for MAE

Paper Section:
    Section 4.1 Evolution of token embeddings across depth

Reproduces:
    1. Figure 2 - Each dot shows the mean attention distance across images for one head (MAE) (with 0 masking)
    2. SAM results (with 0.75 masking)

Usage:
    python analysis/attn_maps.py

Output:
    final/mae_attn_dist_mean_across_imgs.png
    final/sam_attention_boxplots.png

"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
data_dir_class3 = "ImageNet/n03781244"     # monastery
data_dir_class4 = "ImageNet/n02105641"     # 229, old sheep dog
data_dir_class5 = "ImageNet/n02782093"     # 417, balloon
data_dir_class6 = "ImageNet/n02788148"     # 421, bannister
data_dir_class7 = "ImageNet/n02802426"    # 430, basketball
data_dir_class8 = "ImageNet/n03788195"     # 668, mosque
data_dir_class9 = "ImageNet/n04065272"     # 757, RV
data_dir_class10 = "ImageNet/n04273569"    # 814, speedboat

# Collect image paths
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

processed_images_class1 = preprocess_images(image_paths_class1)    # 50 images in each class
processed_images_class2 = preprocess_images(image_paths_class2)
processed_images_class3 = preprocess_images(image_paths_class3)
processed_images_class4 = preprocess_images(image_paths_class4)
processed_images_class5 = preprocess_images(image_paths_class5)
processed_images_class6 = preprocess_images(image_paths_class6)
processed_images_class7 = preprocess_images(image_paths_class7)
processed_images_class8 = preprocess_images(image_paths_class8)
processed_images_class9 = preprocess_images(image_paths_class9)
processed_images_class10 = preprocess_images(image_paths_class10)

processed_images_all = processed_images_class1 + processed_images_class2 + processed_images_class3 + processed_images_class4 + processed_images_class5 + processed_images_class6 + processed_images_class7 + processed_images_class8 + processed_images_class9 + processed_images_class10
print("Batch shape (total images):", np.shape(processed_images_all))  # Should be (150, 224, 224, 3)

# SAM
data_dir_sam = "train"                   
image_paths_sam = [os.path.join(data_dir_sam, img) for img in os.listdir(data_dir_sam) if img.endswith(('.png', '.JPEG', '.jpeg', '.jpg'))]
sampled_paths_sam = image_paths_sam[:100]
processed_images_sam = preprocess_images(sampled_paths_sam)            
print("SAM Batch shape: ", np.shape(processed_images_sam))

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

chkpt_dir = 'mae_pretrain_vit_base.pth'       
model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
print('Model loaded.')

# Attention distance analysis per layer
def check_attention_sum(attn_maps):
    """
    Check if each row in the attention matrix sums to 1.

    """
    row_sums = attn_maps.sum(axis=-1)  # Sum across last axis (columns)
    
    # Compute deviations from expected sum (1.0)
    mean_sum = np.mean(row_sums)
    std_dev = np.std(row_sums)
    
    print(f"Mean row sum across all layers: {mean_sum:.10f}")
    print(f"Standard deviation of row sums: {std_dev:.10f}")

    # Check if all row sums are close to 1
    if np.allclose(row_sums, 1, atol=1e-6):
        print(" Attention maps are correctly normalized (each row sums to ~1).")
    else:
        print(" Attention maps are NOT correctly normalized!")

def normalize_attention(attn_maps):    # if sum not 1
    """
    Normalize each row of the attention matrix so that it sums to 1.

    Args:
        attn_maps (np.array): Raw attention matrix of shape (12, 196, 196)

    Returns:
        np.array: Normalized attention matrix with the same shape.
    """
    row_sums = attn_maps.sum(axis=-1, keepdims=True)  # Sum across each row
    attn_maps_normalized = attn_maps / (row_sums)  # Avoid division by zero
    return attn_maps_normalized

def run_one_image(img, model, mask_ratio=0.75):    # PLOTS first 5 visible patches + return attention maps
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    _, y, mask, _, layer_embeddings, attn_maps, queries, keys, values = model(x.float(), mask_ratio=mask_ratio, return_attention=True)

    layer_embeddings = [layer.squeeze(0).detach().cpu().numpy() for layer in layer_embeddings]
    layer_embeddings_np = np.array(layer_embeddings)

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

    attn_maps_avg = attn_maps_np.mean(axis=1) 
    return attn_maps_avg[:,1:,1:], attn_maps_np, mask_indices, queries_np, keys_np, values_np, layer_embeddings_np, im_masked[0]

# Analyzing mean attn distances per head per layer for a multiple images, taking mean
grid_size = 14  
num_patches = grid_size * grid_size 
patch_positions = []
for i in range(num_patches):
    x = i // grid_size
    y = i % grid_size
    patch_positions.append((x, y))
patch_positions = np.array(patch_positions)   # shape: (196, 2)   unnormalized
print(patch_positions[:5])    
num_patches = patch_positions.shape[0]  # 196
dx = np.abs(patch_positions[:, 0].reshape(num_patches, 1) - patch_positions[:, 0].reshape(1, num_patches))
dy = np.abs(patch_positions[:, 1].reshape(num_patches, 1) - patch_positions[:, 1].reshape(1, num_patches))
# L1
distance_matrix_l1 = dx + dy
np.set_printoptions(precision=4, linewidth=120, suppress=True)
print(distance_matrix_l1)
print(distance_matrix_l1[0, 1])  
print(distance_matrix_l1[0, 100]) 
print(distance_matrix_l1[0, 195])
# L2
distance_matrix_l2 = np.sqrt(dx**2 + dy**2)     # shape: (196, 196)
np.set_printoptions(precision=4, linewidth=120, suppress=True)
print(distance_matrix_l2)
print(distance_matrix_l2[0, 1])   # distance between patch 0 and patch 1 (expected 1.0 for adjacent patches)
print(distance_matrix_l2[0, 195])

def compute_mean_attention_distance(attn_mapss, distance_matrix_l2):
    """
    Compute mean attention distance for each (layer, head)
    attn_mapss: (12, 12, 196, 196) normalized attention maps
    distance_matrix_l2: (196, 196)
    Returns: mean_dist_per_layer_head (12, 12)
    """
    num_layers, num_heads, _, _ = attn_mapss.shape
    mean_distances_per_layer_head = []

    for L in range(num_layers):
        layer_distances = []
        for H in range(num_heads):
            weighted_dist = distance_matrix_l2 * attn_mapss[L, H]  # (196,196)
            mean_dist_per_patch = weighted_dist.sum(axis=1)        # (196,)
            layer_distances.append(mean_dist_per_patch)
        mean_distances_per_layer_head.append(layer_distances)

    mean_distances_per_layer_head = np.array(mean_distances_per_layer_head)  # (12,12,196)
    mean_dist_per_layer_head = mean_distances_per_layer_head.mean(axis=-1)   # (12,12)
    return mean_dist_per_layer_head

def run_multiple_images(images, model, mask_ratio=0.0):     # masking 0 pe ViT vs MAE wala plot
    """
    Run multiple images through MAE, compute attention distances per image,
    then take the mean across images for each head-layer.
    """
    all_mean_distances = []

    for idx, img in enumerate(images):
        print(f"\nProcessing image {idx+1}/{len(images)}")
        _, attn_mapss, mask_indices, queries, keys, values, layer_embeddings_np, masked_img = run_one_image(
            img, model, mask_ratio=mask_ratio
        )
        attn_mapss_norma = normalize_attention(attn_mapss[:, :, 1:, 1:])  # (12,12,196,196)
        mean_dist_per_layer_head = compute_mean_attention_distance(attn_mapss_norma, distance_matrix_l2)
        all_mean_distances.append(mean_dist_per_layer_head)

    all_mean_distances = np.stack(all_mean_distances)  # (N_images, 12, 12)
    mean_over_images = np.mean(all_mean_distances, axis=0)  # (12,12)
    std_over_images = np.std(all_mean_distances, axis=0)    # (12,12)
    return mean_over_images, std_over_images

def plot_attention_distance(mean_dist, patch_size=16, savepath=None):
    
    num_layers, num_heads = mean_dist.shape
    layers = np.arange(1, num_layers + 1)
    mean_dist_pixel = mean_dist * patch_size

    plt.figure(figsize=(8, 6))
    for head in range(num_heads):
        plt.scatter(layers, mean_dist_pixel[:, head], label=f'Head {head + 1}', s=50)

    plt.ylim(0, 125)
    plt.xticks(range(1, num_layers + 1), fontsize=18)  # show all 12 layers
    plt.yticks(fontsize=18) 
    plt.xlabel("Network depth (Layer)", fontsize=20)
    plt.ylabel("Mean attention distance (pixels)", fontsize=20)
    # plt.title("Mean Attention Distance Across Layers for Each Head", fontsize=14)
    plt.legend(title="Heads", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)      # thickness of axes lines
        spine.set_color('black')    # solid black axes

    # Also make tick marks bold black
    ax.tick_params(width=2, color='black')

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

mean_dist_all, std_dist_all = run_multiple_images(processed_images_all, model_mae, mask_ratio=0.0)
plot_attention_distance(mean_dist_all, patch_size=16, savepath="final/mae_attn_dist_mean_across_imgs.png")

# SAM results
def compute_masked_distances(img, model, mask_ratio=0.75):
    _, attn_maps_np, mask_indices, _, _, _, _, _ = run_one_image(img, model, mask_ratio=mask_ratio)

    # attn_maps_np has shape (12,12,N_vis,N_vis)
    grid_size = 14
    patch_positions = np.array([ (i//grid_size, i%grid_size) 
                                 for i in range(grid_size*grid_size) ])
    vis_positions = patch_positions[mask_indices]   # (N_vis,2)
    num_vis = vis_positions.shape[0]
    dx = np.abs(vis_positions[:,0,None] - vis_positions[None,:,0])
    dy = np.abs(vis_positions[:,1,None] - vis_positions[None,:,1])
    D_vis = dx + dy  # (N_vis, N_vis)
    D_vis_l2 = np.sqrt(dx**2 + dy**2)

    attn_norm = normalize_attention(attn_maps_np[:,:,1:,1:])  # (12,12,N_vis,N_vis)   #ig necessary
    mean_dist = np.zeros((12,12), dtype=float)  # layers × heads
    for L in range(12):
        for H in range(12):
            w = attn_norm[L, H]           # (N_vis, N_vis)
            weighted = D_vis * w          # (N_vis, N_vis)
            mean_per_patch = weighted.sum(axis=1)   # (N_vis,)
            mean_dist[L, H] = mean_per_patch.mean() # one value per head
    return mean_dist  # (12,12)

def compute_stats_for_image(img, model, mask_ratio=0.75, runs=100):
    all_dists = []
    patch_size = 16
    for _ in range(runs):
        dist = compute_masked_distances(img, model, mask_ratio=mask_ratio)
        all_dists.append(dist)
    all_dists = np.stack(all_dists, axis=0)  # (runs, 12, 12)
    return all_dists * patch_size

all_images_stats = []  # (100, 100, 12, 12)
for i, img in enumerate(processed_images_sam):
    print(f"Processing image {i+1}/100")
    dist_stack = compute_stats_for_image(img, model_mae, mask_ratio=0.75, runs=1)
    all_images_stats.append(dist_stack)
all_images_stats = np.stack(all_images_stats, axis=0)  # Shape: (100, 1, 12, 12)

# Plotting box-plots SAM
all_dists = all_images_stats[:, 0]
num_layers = 12
num_heads = 12
fig, axes = plt.subplots(3, 4, figsize=(24, 16))
axes = axes.flatten()
for L in range(num_layers):
    ax = axes[L]
    # Collect 100 values / head
    layer_data = all_dists[:, L, :]    # shape (100, 12)
    headwise_vals = [ layer_data[:, H] for H in range(num_heads) ]
    ax.boxplot(
        headwise_vals,
        showfliers=True,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(color='black', linewidth=2))
    ax.set_title(f"Layer {L+1}", fontsize=14)
    ax.set_xticks(range(1, num_heads + 1))
    ax.set_xticklabels(range(1, num_heads + 1), fontsize=10)
    ax.set_xlabel("Head", fontsize=11)
    if L % 4 == 0:
        ax.set_ylabel("Attention distance (pixels)", fontsize=12)
    else:
        ax.set_ylabel("")  
plt.subplots_adjust( hspace=0.45, wspace=0.25)   
# plt.savefig("final/sam_attention_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()
