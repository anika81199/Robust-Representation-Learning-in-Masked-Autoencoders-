"""
Finetune only the linear classification head of a pretrained MAE encoder on any
custom dataset. The full backbone (patch_embed, cls_token, pos_embed, transformer
blocks, fc_norm) is frozen; only the final nn.Linear head is updated.

Split strategy: fixed number of images per class for train / val / test.
This ensures every class contributes equally regardless of how many images it has,
avoiding the accuracy bias that comes from class-size imbalance.

Defaults are set for Caltech-256 (256_ObjectCategories):
    256_ObjectCategories/
      001.ak47/001_0001.jpg  ...
      002.american-flag/...
      ...  (258 class folders, 80–827 images each)

  Caltech-256 min class size = 80, so defaults are train=60 / val=10 / test=10.

Usage (Caltech-256, all defaults):
  python finetune_head_custom_dataset.py

Usage (custom dataset or different split sizes):
  python finetune_head_custom_dataset.py \
      --data_dir /path/to/dataset \
      --checkpoint /path/to/checkpoint.pth \
      --train_per_class 60 --val_per_class 10 --test_per_class 10 \
      --epochs 20 --lr 1e-3 --batch_size 32
"""

import argparse
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import models_mae

class MAEClassifier(nn.Module):
    """MAE encoder + global-average-pool head (same forward as existing code)."""

    def __init__(self, mae_model, embed_dim=768, num_classes=10, mask_ratio=0.0):
        super().__init__()
        self.patch_embed    = mae_model.patch_embed
        self.cls_token      = mae_model.cls_token
        self.pos_embed      = mae_model.pos_embed
        self.blocks         = mae_model.blocks
        self.fc_norm        = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head           = nn.Linear(embed_dim, num_classes)
        self.mask_ratio     = mask_ratio
        self.random_masking = mae_model.random_masking

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, _, _ = self.random_masking(x, self.mask_ratio)
        cls_token  = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].mean(dim=1)   # global average pool (no cls token)
        return self.head(self.fc_norm(x))

    def freeze_backbone(self):
        """Freeze every parameter except the linear head."""
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith('head.')
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {n_trainable:,} / {n_total:,}  ({100 * n_trainable / n_total:.3f}%)")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def get_transform(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImagePathDataset(Dataset):
    """Lightweight dataset built from explicit (path, label) pairs."""

    def __init__(self, samples: list, transform):
        self.samples   = samples   # list of (path_str, int_label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def build_fixed_split_loaders(data_dir: str,
                               train_per_class: int,
                               val_per_class: int,
                               test_per_class: int,
                               batch_size: int,
                               num_workers: int,
                               seed: int = 42):
    """
    For each class folder, shuffle images with a fixed seed then assign:
      first train_per_class  → train
      next  val_per_class    → val
      next  test_per_class   → test

    Classes with fewer images than (train+val+test) are skipped with a warning.
    """
    needed = train_per_class + val_per_class + test_per_class

    # Collect sorted class folders (mirrors ImageFolder's alphabetical ordering)
    class_dirs = sorted([
        d for d in os.scandir(data_dir) if d.is_dir()
    ], key=lambda e: e.name)

    class_names = [d.name for d in class_dirs]
    num_classes = len(class_names)

    train_samples, val_samples, test_samples = [], [], []
    skipped = []

    rng = random.Random(seed)

    for label, entry in enumerate(class_dirs):
        imgs = sorted([
            os.path.join(entry.path, f)
            for f in os.listdir(entry.path)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ])

        if len(imgs) < needed:
            skipped.append((entry.name, len(imgs)))
            continue

        rng.shuffle(imgs)
        train_samples += [(p, label) for p in imgs[:train_per_class]]
        val_samples   += [(p, label) for p in imgs[train_per_class:train_per_class + val_per_class]]
        test_samples  += [(p, label) for p in imgs[train_per_class + val_per_class:needed]]

    if skipped:
        print(f"\nWarning: {len(skipped)} class(es) skipped — fewer than {needed} images:")
        for name, cnt in skipped:
            print(f"  {name}: {cnt} images")

    # Adjust num_classes if any were skipped
    included_labels = sorted({lbl for _, lbl in train_samples})
    # Remap labels to be contiguous [0, N) in case any were skipped
    label_map = {old: new for new, old in enumerate(included_labels)}
    train_samples = [(p, label_map[l]) for p, l in train_samples]
    val_samples   = [(p, label_map[l]) for p, l in val_samples]
    test_samples  = [(p, label_map[l]) for p, l in test_samples]
    class_names   = [class_names[l] for l in included_labels]
    num_classes   = len(class_names)

    train_ds = ImagePathDataset(train_samples, get_transform(True))
    val_ds   = ImagePathDataset(val_samples,   get_transform(False))
    test_ds  = ImagePathDataset(test_samples,  get_transform(False))

    print(f"\nSplit (fixed {train_per_class}/{val_per_class}/{test_per_class} per class):")
    print(f"  Train : {len(train_ds):>6} images  ({num_classes} classes × {train_per_class})")
    print(f"  Val   : {len(val_ds):>6} images  ({num_classes} classes × {val_per_class})")
    print(f"  Test  : {len(test_ds):>6} images  ({num_classes} classes × {test_per_class})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes, class_names


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def build_model(checkpoint_path: str, num_classes: int,
                embed_dim: int = 768, mask_ratio: float = 0.0) -> MAEClassifier:
    mae_model  = getattr(models_mae, 'mae_vit_base_patch16')()
    classifier = MAEClassifier(mae_model, embed_dim=embed_dim,
                               num_classes=num_classes, mask_ratio=mask_ratio)

    if checkpoint_path:
        ckpt       = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)

        # Drop head keys — shape [1000, 768] won't match the new num_classes
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}

        msg = classifier.load_state_dict(state_dict, strict=False)
        print(f"\nLoaded checkpoint: {checkpoint_path}")
        print(f"  Missing (randomly initialised): {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  Unexpected (ignored):          {msg.unexpected_keys}")

    return classifier


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds    = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total if total > 0 else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, num_classes, class_names = build_fixed_split_loaders(
        data_dir        = args.data_dir,
        train_per_class = args.train_per_class,
        val_per_class   = args.val_per_class,
        test_per_class  = args.test_per_class,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
    )

    model = build_model(args.checkpoint, num_classes,
                        embed_dim=args.embed_dim, mask_ratio=args.mask_ratio)
    model.freeze_backbone()
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt    = os.path.join(args.output_dir, 'best_head.pth')

    print(f"\nTraining for {args.epochs} epochs (head only)...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct      += (logits.argmax(dim=1) == labels).sum().item()
            total        += labels.size(0)

        scheduler.step()

        train_acc = correct / total
        val_acc   = evaluate(model, val_loader, device)
        avg_loss  = running_loss / total

        print(f"Epoch {epoch:3d}  loss {avg_loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':       epoch,
                'model':       model.state_dict(),
                'val_acc':     val_acc,
                'class_names': class_names,
                'num_classes': num_classes,
            }, best_ckpt)
            print(f"         --> saved best model ({val_acc:.4f}) → {best_ckpt}")

    # ── Final test evaluation on the held-out test set ────────────────────────
    print(f"\nLoading best checkpoint for test evaluation...")
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state['model'])
    test_acc = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Checkpoint        : {best_ckpt}")
    print(f"{'='*50}")


CALTECH256_DIR = '256_ObjectCategories'
MAE_CHECKPOINT = 'mae_finetuned_vit_base.pth'


def get_args():
    p = argparse.ArgumentParser('MAE head-only finetuning — fixed per-class split')
    p.add_argument('--data_dir',         default=CALTECH256_DIR)
    p.add_argument('--checkpoint',       default=MAE_CHECKPOINT)

    # Split sizes — defaults safe for Caltech-256 (min class = 80 images)
    p.add_argument('--train_per_class',  type=int, default=60,
                   help='Images per class used for training.')
    p.add_argument('--val_per_class',    type=int, default=10,
                   help='Images per class used for validation.')
    p.add_argument('--test_per_class',   type=int, default=10,
                   help='Images per class used for held-out testing.')

    p.add_argument('--seed',             type=int,   default=42,
                   help='Random seed for the per-class shuffle (keeps split reproducible).')
    p.add_argument('--embed_dim',        type=int,   default=768,
                   help='768 for ViT-B, 1024 for ViT-L.')
    p.add_argument('--mask_ratio',       type=float, default=0.0)
    p.add_argument('--epochs',           type=int,   default=10)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--weight_decay',     type=float, default=0.05)
    p.add_argument('--batch_size',       type=int,   default=32)
    p.add_argument('--num_workers',      type=int,   default=4)
    p.add_argument('--output_dir',       default='./caltech256_head_finetune')
    return p.parse_args()


if __name__ == '__main__':
    train(get_args())
