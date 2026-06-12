""" 
Purpose:
    Reproduce the t-SNE visualizations of CLS and patch tokens

Paper Section:
    Section 4.1 Evolution of token embeddings across depth

Reproduces:
    Figure 1 t-SNE visualizations of token embeddings across encoder layers
    (a) CLS tokens and (b) Mean patch tokens

Usage:
    python tsne_visualizations.py

Output:
    final/tsne_class_tokens.png
    final/tsne_mean_patch_tokens.png

"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

np.float = float

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(img, title=''):
    # image is [H, W, 3]
    assert img.shape[2] == 3
    image = torch.tensor(img)
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    # plt.imshow(torch.clip((image*255), 0 , 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    return

# Preprocess images
def preprocess_images(image_paths):
    processed_images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
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

processed_images_all = [processed_images_class1 , processed_images_class2 , processed_images_class3 , processed_images_class4 , processed_images_class5 , processed_images_class6 , processed_images_class7 , processed_images_class8 , processed_images_class9 , processed_images_class10]
print(np.shape(processed_images_all))
print(len(processed_images_all))

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    chkpt_dir='mae_finetuned_vit_base.pth'
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, mask_ratio=0.75):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    _, y, mask, latent, layer_embeddings = model(x.float(), mask_ratio=mask_ratio)
    latent_tensor = latent.squeeze(0)               # (50x768)
    print("Matrix shape:", latent_tensor.shape)     

    reduced_matrix = latent_tensor[1:, :]         # (49, 768)
    print("Reduced matrix shape:", reduced_matrix.shape)
    reduced_matrix = reduced_matrix.detach().numpy()

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

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")
    plt.show()
    return reduced_matrix, mask_indices

def extract_latent(images, model, mask_ratio = 0.75):
    latent_tensors = []     # class tokens --> (1,768)
    matrices=[]             # latent shape --> (50, 768)    
    reduced = []            # patch tokens --> (49, 768)
    positions = []          # of visible patches --> (10, 49)
    layerwise_embeddings = []  # Store embeddings from each layer

    for idx, img in enumerate(images):
        print(f"\nProcessing Image {idx + 1}/{len(images)}")
        x = torch.tensor(img)  
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)  

        # Run MAE model
        with torch.no_grad():
            loss, y, mask, latent, layer_embeddings = model(x.float(), mask_ratio=mask_ratio)   # change masking ratio and check allignment
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
        latent_tensors.append(class_token.squeeze(0))  #  [1, 768]
        matrices.append(matrix_numpy)         # 50x768 stored for each images
        reduced.append(reduced_matrix)
        positions.append(mask_indices)  # Indices of visible patches
    latent_tensors = torch.stack(latent_tensors)  # Final shape: [N, 768]
    return latent_tensors, matrices, reduced, positions, layerwise_embeddings     # tensor, list

chkpt_dir = 'mae_pretrain_vit_base.pth'      
model_mae = prepare_model(chkpt_dir, arch='mae_vit_base_patch16')
print('Model loaded.')

"""
t-SNE for 10 classes
"""

layerwise_embeddings_list = []
for i, proc_imgs in enumerate(processed_images_all, start=1):
    latent_tensors, matrices, reduced, positions, layerwise_embeddings = extract_latent(proc_imgs, model_mae)
    print(f"[Class {i}] class tokens shape:", np.shape(latent_tensors))
    print(f"[Class {i}] patch tokens shape:", np.shape(reduced))
    print(f"[Class {i}] positions shape:", np.shape(positions))
    print(f"[Class {i}] layerwise_embedding shape:", np.shape(layerwise_embeddings))
    layerwise_embeddings = layerwise_embeddings[:10]
    layerwise_embeddings_list.append(layerwise_embeddings)

# Stack: (num_classes*10, 12, 50, 768) assuming 1 CLS + 49 patches
tokens_all = np.vstack(layerwise_embeddings_list)
print("tokens_all shape:", tokens_all.shape)

# Labels: 10 images per class, 10 classes => 100 samples
num_classes = len(processed_images_all)
samples_per_class = 10
labels = np.repeat(np.arange(num_classes), samples_per_class)

# ---------------------------
# t-SNE for CLS tokens (layer-wise 3x4 grid)
# ---------------------------
class_tokens = tokens_all[:, :, 0, :]  # (100, 12, 768)
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
for layer in range(12):
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, metric='euclidean')
    emb2d = tsne.fit_transform(class_tokens[:, layer, :])  # Shape: (100, 2)
    r, c = divmod(layer, 4)
    ax = axes[r, c]
    scatter = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, cmap="tab10", alpha=0.85, s=24)
    ax.set_title(f"Layer {layer+1}")
    ax.set_xticks([]); ax.set_yticks([])

handles, _ = scatter.legend_elements()
legend_labels = [f"Class {i+1}" for i in range(num_classes)]
fig.legend(handles[:num_classes], legend_labels, loc="lower center", ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
# plt.savefig("final/tsne_class_tokens.png", dpi=300, bbox_inches='tight')
plt.show() 

# ---------------------------
# t-SNE for mean patch tokens per image (layer-wise)
# ---------------------------
patch_tokens = tokens_all[:, :, 1:, :]              # (100, 12, 49, 768)
mean_patch_tokens = patch_tokens.mean(axis=2)       # (100, 12, 768)
mean_patch_tokens_std = np.empty_like(mean_patch_tokens)
scaler = StandardScaler()  
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

for layer in range(12):
    X = mean_patch_tokens[:, layer, :]          # shape (100, 768)
    mean_patch_tokens_std[:, layer, :] = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate='auto',metric='euclidean', max_iter=1000)
    emb2d = tsne.fit_transform(mean_patch_tokens_std[:, layer, :])  # (100, 2)
    r, c = divmod(layer, 4)
    ax = axes[r, c]
    scatter = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, cmap="tab10", alpha=0.85, s=24)
    ax.set_title(f"Layer {layer+1}")
    ax.set_xticks([]); ax.set_yticks([])
handles, _ = scatter.legend_elements()
fig.legend(handles[:num_classes], legend_labels, loc="lower center", ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
# plt.savefig("final/tsne_mean_patch_tokens.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------
# t-SNE on ALL 49 patch tokens per image
# ---------------------------
labels_patches = np.repeat(np.arange(num_classes), samples_per_class * 49)
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
fig.suptitle("t-SNE of All Patch Tokens Across Layers (49 per image)", fontsize=16)
for layer in range(12):
    layer_pt = patch_tokens[:, layer, :, :].reshape(-1, 768)  # (100*49, 768) = (4900, 768)
    layer_pt = scaler.fit_transform(layer_pt)
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate='auto', metric='euclidean', max_iter=1000)
    emb2d = tsne.fit_transform(layer_pt)
    r, c = divmod(layer, 4)
    ax = axes[r, c]
    scatter = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=labels_patches, cmap="tab10", alpha=0.6, s=4)
    ax.set_title(f"Layer {layer+1}")
    ax.set_xticks([]); ax.set_yticks([])
handles, _ = scatter.legend_elements()
fig.legend(handles[:num_classes], legend_labels, loc="lower center", ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
# plt.savefig("final/tsne_patch_tokens.png", dpi=300, bbox_inches='tight')
plt.show()
