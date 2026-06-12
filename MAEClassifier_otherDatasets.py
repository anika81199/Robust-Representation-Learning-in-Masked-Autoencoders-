""" 
Purpose:
    Reproduce the classification accuracy results for IMAGENET A, R, C datasets

Paper Section:
    Section 4.2 Robustness of fine-tuned MAE under input perturbations

Usage:
    python analysis/MAEClassifier_otherDatasets.py

Output:
    Mean accuracy
"""

import torch.nn as nn
import sys
import os
import requests
import hashlib
import statistics
import math
import torch
import numpy as np
import cv2
import seaborn as sns
from tqdm import tqdm
import random
import pandas as pd
import ast
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
import models_mae 
from collections import defaultdict 
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import math
from tqdm import tqdm
import json


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
        x, _, _ = self.random_masking(x, self.mask_ratio)

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

# def load_imagenet_labels(label_file_path):
#     with open(label_file_path) as f:
#         return [line.strip() for line in f.readlines()]

print("\n================ Imagenet-R ACCURACY (200 classes) ================\n")

# with open("imagenet_class_index.json", "r") as f:
#     class_index = json.load(f)
# # wnid -> imagenet-1k index
# wnid_to_class_id = {
#     v[0]: int(k)
#     for k, v in class_index.items()
# }

# IMAGENET_R_ROOT = "imagenet-r"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# global_correct = 0
# global_total = 0
# per_class_acc = {}

# -------------------------------------------------------
# Iterate over ImageNet-R folders
# -------------------------------------------------------
# for wnid in sorted(os.listdir(IMAGENET_R_ROOT)):

#     class_dir = os.path.join(IMAGENET_R_ROOT, wnid)
#     if not os.path.isdir(class_dir):
#         continue
    
#     if wnid not in wnid_to_class_id:
#         print(f"Skipping unknown WNID: {wnid}")
#         continue

#     class_id = wnid_to_class_id[wnid]
#     print(f"\n----- Class {class_id}: {wnid} -----")

#     img_paths = [
#         os.path.join(class_dir, img)
#         for img in os.listdir(class_dir)
#         if img.lower().endswith(('.jpg', '.jpeg', '.png'))
#     ]

#     if len(img_paths) == 0:
#         continue

#     processed_images = preprocess_images(img_paths)
#     batch_tensor = torch.stack(processed_images).to(device)

#     true_labels = torch.full(
#         (len(img_paths),),
#         class_id,
#         dtype=torch.long,
#         device=device
#     )

#     with torch.no_grad():
#         logits = classifier(batch_tensor)
#         preds = logits.argmax(dim=1)

#     correct = (preds.cpu().numpy() == np.array(true_labels)).sum()
#     acc = correct / len(img_paths)

#     per_class_acc[wnid] = acc
#     global_correct += correct
#     global_total += len(img_paths)

#     print(f"Top-1 Accuracy: {acc:.4f} ({correct}/{len(img_paths)})")

# mean_acc = global_correct / global_total
# macro_acc = np.mean(list(per_class_acc.values()))

# print("\n===========================================================")
# print(f" Global Accuracy (micro): {mean_acc:.4f}") = 34.81
# print(f" Mean Class Accuracy (macro): {macro_acc:.4f}") = 36.54
# print("===========================================================\n")


print("\n================ Imagenet-C ACCURACY (10 classes) ================\n")

# with open("imagenet_class_index.json", "r") as f:
#     class_index = json.load(f)
# wnid_to_class_id = {
#     v[0]: int(k)
#     for k, v in class_index.items()
# }
# IMAGENET_C_ROOT = "Imagenet-C"

# SELECTED_CLASSES = {
#     'n02106662', 'n01855032', 'n02107312', 'n02105641',
#     'n02782093', 'n02788148', 'n02802426',
#     'n03788195', 'n04065272', 'n04273569'
# }

# SEVERITIES = [1, 2, 3, 4, 5]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for corruption in sorted(os.listdir(IMAGENET_C_ROOT)):

#     corruption_dir = os.path.join(IMAGENET_C_ROOT, corruption)
#     if not os.path.isdir(corruption_dir):
#         continue

#     print("\n===========================================================")
#     print(f" Corruption: {corruption}")
#     print("===========================================================")

#     severity_acc = {}
#     corruption_correct = 0
#     corruption_total = 0

#     # ---------------------------------------------------
#     # Loop over severities
#     # ---------------------------------------------------
#     for severity in SEVERITIES:

#         severity_dir = os.path.join(corruption_dir, str(severity))
#         if not os.path.isdir(severity_dir):
#             continue

#         severity_correct = 0
#         severity_total = 0

#         for wnid in SELECTED_CLASSES:

#             class_dir = os.path.join(severity_dir, wnid)
#             if not os.path.isdir(class_dir):
#                 continue

#             class_id = wnid_to_class_id[wnid]

#             img_paths = [
#                 os.path.join(class_dir, img)
#                 for img in os.listdir(class_dir)
#                 if img.lower().endswith(('.jpg', '.jpeg', '.png'))
#             ]

#             if len(img_paths) == 0:
#                 continue

#             processed_images = preprocess_images(img_paths)
#             batch_tensor = torch.stack(processed_images).to(device)

#             true_labels = torch.full(
#                 (len(img_paths),),
#                 class_id,
#                 dtype=torch.long,
#                 device=device
#             )

#             with torch.no_grad():
#                 logits = classifier(batch_tensor)
#                 preds = logits.argmax(dim=1)

#             correct = (preds == true_labels).sum().item()

#             severity_correct += correct
#             severity_total += len(img_paths)

#         if severity_total > 0:
#             acc = severity_correct / severity_total
#             severity_acc[severity] = acc

#             corruption_correct += severity_correct
#             corruption_total += severity_total

#             print(f"Severity {severity}: Mean Accuracy = {acc:.4f}")

#     # ---------------------------------------------------
#     # Mean across all severities for this corruption
#     # ---------------------------------------------------
#     if corruption_total > 0:
#         mean_corruption_acc = corruption_correct / corruption_total
#         print("-----------------------------------------------------------")
#         print(f"Mean Accuracy across all severities ({corruption}): "
#               f"{mean_corruption_acc:.4f}")
        

print("\n================ Imagenet-A ACCURACY (200 classes) ================\n")

with open("imagenet_class_index.json", "r") as f:
    class_index = json.load(f)
# wnid -> imagenet-1k index
wnid_to_class_id = {
    v[0]: int(k)
    for k, v in class_index.items()
}

IMAGENET_A_ROOT = "imagenet-a"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_correct = 0
global_total = 0
per_class_acc = {}

# -------------------------------------------------------
# Iterate over ImageNet-A folders
# -------------------------------------------------------
for wnid in sorted(os.listdir(IMAGENET_A_ROOT)):

    class_dir = os.path.join(IMAGENET_A_ROOT, wnid)
    if not os.path.isdir(class_dir):
        continue
    
    if wnid not in wnid_to_class_id:
        print(f"Skipping unknown WNID: {wnid}")
        continue

    class_id = wnid_to_class_id[wnid]
    print(f"\n----- Class {class_id}: {wnid} -----")

    img_paths = [
        os.path.join(class_dir, img)
        for img in os.listdir(class_dir)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if len(img_paths) == 0:
        continue

    processed_images = preprocess_images(img_paths)
    batch_tensor = torch.stack(processed_images).to(device)

    true_labels = torch.full(
        (len(img_paths),),
        class_id,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        logits = classifier(batch_tensor)
        preds = logits.argmax(dim=1)

    correct = (preds.cpu().numpy() == np.array(true_labels)).sum()
    acc = correct / len(img_paths)

    per_class_acc[wnid] = acc
    global_correct += correct
    global_total += len(img_paths)

    print(f"Top-1 Accuracy: {acc:.4f} ({correct}/{len(img_paths)})")

mean_acc = global_correct / global_total
macro_acc = np.mean(list(per_class_acc.values()))

print("\n===========================================================")
print(f" Global Accuracy (micro): {mean_acc:.4f}") 
print(f" Mean Class Accuracy (macro): {macro_acc:.4f}")
print("===========================================================\n")