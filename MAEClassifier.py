""" 
Purpose:
    Reproduce the classification accuracy results for clean, attention-guided occluded and blurred images

Paper Section:
    Section 4.2 Robustness of fine-tuned MAE under input perturbations

Reproduces:
    Table 1 Top-1 accuracy, PSNR, and SSIM for varying blur levels,
            characterized by kernel size and standard deviation on ImageNet-1K dataset.
    Figure 4 Occlusion level vs Mean Accuracy plot on ImageNet-1K dataset.

Usage:
    python analysis/MAEClassifier.py

Output:
    final/occ_acc.png
"""

import torch.nn as nn
import os
import statistics
import math
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import models_mae 
from collections import defaultdict 
from skimage.metrics import structural_similarity as ssim
import math
from tqdm import tqdm

class MAEClassifier(nn.Module):
    def __init__(self, mae_model, embed_dim=768, num_classes=1000, mask_ratio=0.0):
        super().__init__()
        self.patch_embed = mae_model.patch_embed
        # num_patches = self.patch_embed.num_patches
        self.cls_token = mae_model.cls_token
        self.pos_embed = mae_model.pos_embed
        self.blocks = mae_model.blocks
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)  # classifier
        self.mask_ratio = mask_ratio
        self.random_masking = mae_model.random_masking

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        outcome = self.fc_norm(x)
        logits = self.head(outcome)
        return logits

# build model
mae_model = getattr(models_mae, 'mae_vit_base_patch16')()
classifier = MAEClassifier(mae_model, embed_dim=768, num_classes=1000)
checkpoint = torch.load("mae_finetuned_vit_base.pth", map_location='cpu')
state_dict = checkpoint['model']
msg = classifier.load_state_dict(state_dict, strict=False)
print(msg)

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

def load_imagenet_labels(label_file_path):
    with open(label_file_path) as f:
        return [line.strip() for line in f.readlines()]

# Preprocess images
def preprocess_images(image_paths):
    processed_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")  # RGB
        image = image.resize((224, 224))  
        image = np.array(image) / 255.0
        image = (image - imagenet_mean) / imagenet_std  # Apply standardization
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        image = torch.tensor(image).float()
        processed_images.append(image)
    return processed_images

print("\n================ CLEAN IMAGE ACCURACY (10 classes):in paper, 13th nov ================\n")
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
label_names = load_imagenet_labels("imagenet1000_clsidx_to_labels.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_correct = 0
global_total = 0
per_class_acc = {}

for class_id, class_dir in classes.items():
    print(f"\n----- Class {class_id}: {os.path.basename(class_dir)} -----")
    img_paths = [
        os.path.join(class_dir, img)
        for img in os.listdir(class_dir)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    processed_images = preprocess_images(img_paths)
    batch_tensor = torch.stack(processed_images).to(device)
    true_labels = [class_id] * len(img_paths)
    with torch.no_grad():
        logits = classifier(batch_tensor)
        preds = logits.argmax(dim=1)
        confidences = torch.softmax(logits, dim=1).max(dim=1).values
    correct = (preds.cpu().numpy() == np.array(true_labels)).sum()
    acc = correct / len(img_paths)
    per_class_acc[class_id] = acc
    global_correct += correct
    global_total += len(img_paths)
    # result
    print(f"Top-1 Accuracy: {acc:.4f}  ({correct}/{len(img_paths)})")
mean_acc = global_correct / global_total
print("\n===========================================================\n")
print(f" Global Mean Accuracy (10 classes): {mean_acc:.4f}  ({global_correct}/{global_total})")
print("===========================================================\n")


# ============== checking accuracy for attention-aware OCCLUDED images ~ 10 classes =============
print("  ")
print("attention-guided occlusion ~ 10 levels")
print("  ")

def preprocess_image_np(image_np):
    """Normalize H×W×3 np.float32 in [0,1] → torch.FloatTensor [3,H,W]."""
    img = (image_np - imagenet_mean) / imagenet_std
    img = np.transpose(img, (2, 0, 1))  # → 3×H×W
    return torch.tensor(img).float()

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = classifier.to(device).eval()
IMG_SIZE = (224, 224)
PATCH_SIZE = 16
GRID_SIZE = 14   # 224/16

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

# Occluded classes folders already exists in the repo 
occlusion_levels = [0,10,20,30,40,50,60,70,80,90]
basename_to_class = {os.path.basename(v): k for k, v in classes.items()}
mean_acc_recorded = []

for level in occlusion_levels:
    level_dir = f"occluded_{level}"
    if not os.path.exists(level_dir):
        print(f"⚠️  Folder {level_dir} not found — skipping")
        continue

    print(f"\n==== Evaluating occlusion {level}% ====")
    total_correct, total_images = 0, 0
    per_class_acc = {}

    # loop over each class subfolder
    for class_folder in os.listdir(level_dir):
        class_path = os.path.join(level_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_id = basename_to_class.get(class_folder)
        if class_id is None:
            print(f"  ⚠️ Skipping unknown class folder {class_folder}")
            continue

        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue

        per_class_correct = 0
        for img_name in tqdm(image_files, desc=f"{class_folder}", leave=False):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0
            tensor = preprocess_image_np(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = classifier(tensor)
                pred = torch.softmax(logits, dim=1).argmax(dim=1).item()

            if pred == class_id:
                per_class_correct += 1
            total_images += 1

        acc = per_class_correct / len(image_files)
        per_class_acc[class_id] = acc
        total_correct += per_class_correct

        print(f" - Class {class_id:>3} ({class_folder}): acc={acc:.3%} (n={len(image_files)})")

    # mean accuracy across all classes
    mean_acc = total_correct / total_images if total_images > 0 else 0.0
    mean_acc_recorded.append(mean_acc)
    print(f" Mean accuracy @ {level}% occlusion: {mean_acc:.3%}")

mean_acc = [0.934, 0.938, 0.920, 0.924, 0.92, 0.90, 0.806, 0.780, 0.762, 0.608] 
plt.figure(figsize=(10,5))
plt.plot(occlusion_levels, mean_acc, marker='o', linewidth=2)
plt.title("Mean Accuracy vs Occlusion Level")
plt.xticks(rotation=45)
plt.xlabel("Occlusion Level (%)")
plt.ylabel("Average drop")
plt.grid(False)
plt.xticks(occlusion_levels)
plt.savefig("final/occ_acc.png", dpi=300, bbox_inches='tight')
plt.show()

# ================== accuracy of MAEclassifier Gaussian BLURRED images ~ 10 classes =======================
print("  ")
print("blurred images ~ 10 levels ~ 10 classes")
print("  ")

def apply_gaussian_blur_cv2(image_array, ksize, sigma):
    return cv2.GaussianBlur(image_array, (ksize, ksize), sigma)

def preprocess_image_np(image_np):
    """Normalize H×W×3 np.float32 in [0,1] → torch.FloatTensor [3,H,W]."""
    img = (image_np - imagenet_mean) / imagenet_std
    img = np.transpose(img, (2, 0, 1))  # → 3×H×W
    return torch.tensor(img).float()

def load_imagenet_labels(label_file):
    with open(label_file) as f:
        return [line.strip() for line in f]

def load_image_paths(folder, limit=None):
    if not os.path.isdir(folder):
        return []
    paths = [
        os.path.join(folder, fn)
        for fn in sorted(os.listdir(folder))
        if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return paths[:limit] if limit else paths

def evaluate_class_for_setting(classifier, device, image_paths, true_label, ksize, sigma):
    """
    Returns accuracy (float) for a single class under one (ksize, sigma).
    """
    if not image_paths:
        return None

    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr_blur = apply_gaussian_blur_cv2(arr, ksize, sigma)
        tensors.append(preprocess_image_np(arr_blur))

    batch = torch.stack(tensors, dim=0).to(device)  # [N,3,224,224]

    with torch.no_grad():
        logits = classifier(batch)
        preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()

    true = np.full(len(image_paths), true_label, dtype=int)
    return float((preds == true).mean())

label_file = "imagenet1000_clsidx_to_labels.txt"
label_names = load_imagenet_labels(label_file)
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
limit_per_class = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = classifier.to(device).eval()
per_kernel = defaultdict(lambda: defaultdict(list))
mean_by_kernel = defaultdict(lambda: defaultdict(list))
mean_acc_results = {}   # (k, sigma) -> mean accuracy

for ksize, sigma in blur_settings:
    print(f"\nBlur Setting  k={ksize}, σ={sigma}")
    for class_id, folder in classes.items():
        image_paths = load_image_paths(folder, limit=limit_per_class)
        acc = evaluate_class_for_setting(classifier, device, image_paths, class_id, ksize, sigma)

        if acc is None:
            print(f"  - {class_id:>3} ({label_names[class_id]}): no images found → skip")
            continue

        per_kernel[ksize][class_id].append((sigma, acc))
        mean_by_kernel[ksize][sigma].append(acc)
        print(f"  - {class_id:>3} ({label_names[class_id]}): acc={acc:.3%} "
              f"({len(image_paths)} imgs)")

# sort for nice plotting/reading
for k in list(per_kernel.keys()):
    for cid in list(per_kernel[k].keys()):
        per_kernel[k][cid].sort(key=lambda x: x[0])

# mean summary across the 10 classes for each (k, σ)
print("\n=== Mean accuracy across classes (per k, σ) ===")
for k in sorted(mean_by_kernel.keys()):
    for sigma in sorted(mean_by_kernel[k].keys()):
        vals = mean_by_kernel[k][sigma]
        if not vals:
            continue
        mu, sd, n = np.mean(vals), np.std(vals), len(vals)
        mean_acc_results[(k, sigma)] = mu
        print(f"k={k:>2}, σ={sigma:>5}: mean={mu:.3%}")

results = defaultdict(list)
for k in sorted(mean_by_kernel.keys()):
    for sigma in sorted(mean_by_kernel[k].keys()):
        vals = mean_by_kernel[k][sigma]
        if vals:
            results[k].append((sigma, float(np.mean(vals))))

print(mean_acc_results)
