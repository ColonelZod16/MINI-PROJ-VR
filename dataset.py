"""
dataset.py
==========
PyTorch Dataset for multi-label classification.

Key difference from single-label:
  - One image can belong to MULTIPLE categories (item1 + item2 in same photo)
  - Target is a float binary vector [0,1,0,1,0] not a single int
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class FashionMultiLabelDataset(Dataset):
    """
    Reads the CSV manifests produced by dataset_pipeline.py.

    Multi-label logic:
        One physical image (image_stem) may appear multiple times in the CSV
        — once per clothing item annotated in it.
        We group all rows by image_stem and build a binary label vector
        where index=1 for every category present in that image.

    Returns
    -------
    image  : FloatTensor [3, H, W]
    target : FloatTensor [num_classes]  — multi-hot binary vector
    stem   : str  — image filename stem (useful for debugging)
    """

    def __init__(
        self,
        csv_path:     str,
        images_dir:   str,
        category_map: Dict[str, int],    # { "trousers": 0, "skirt": 1, ... }
        transform:    Optional[Callable] = None,
        img_size:     int = 224,
    ):
        self.images_dir  = Path(images_dir)
        self.category_map = category_map
        self.num_classes  = len(category_map)
        self.transform    = transform or get_eval_transforms(img_size)

        # ── Group rows by image_stem → multi-hot label ────────────────
        stem_to_labels: Dict[str, set] = {}
        stem_to_path:   Dict[str, str] = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem     = row["image_stem"]
                cat      = row["category_name"]
                img_path = row.get("image_path", "")

                if cat not in category_map:
                    continue
                if stem not in stem_to_labels:
                    stem_to_labels[stem] = set()
                    stem_to_path[stem]   = img_path

                stem_to_labels[stem].add(category_map[cat])

        # Convert to sorted list for indexing
        self.stems   = sorted(stem_to_labels.keys())
        self.labels  = {s: stem_to_labels[s] for s in self.stems}
        self.paths   = {s: stem_to_path[s]   for s in self.stems}

        print(f"  [Dataset] {Path(csv_path).name}  →  "
              f"{len(self.stems):,} unique images  |  "
              f"{sum(len(v) for v in self.labels.values()):,} total labels")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        stem  = self.stems[idx]

        # ── Load image ────────────────────────────────────────────────
        img_path = self.images_dir / self.paths[stem]
        if not img_path.exists():
            # fallback: search by stem
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                p = self.images_dir / (stem + ext)
                if p.exists():
                    img_path = p
                    break

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # ── Build multi-hot target vector ─────────────────────────────
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label_idx in self.labels[stem]:
            target[label_idx] = 1.0

        return img, target, stem

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weight for WeightedRandomSampler.
        Samples containing rarer classes get higher weight.
        Uses the *rarest* class in each image as the sample weight.
        """
        # Count how many samples contain each class
        class_counts = torch.zeros(self.num_classes)
        for label_set in self.labels.values():
            for c in label_set:
                class_counts[c] += 1

        class_weights = 1.0 / (class_counts + 1e-8)

        sample_weights = []
        for stem in self.stems:
            label_set = self.labels[stem]
            # Weight = max class weight among all labels in this sample
            w = max(class_weights[c].item() for c in label_set) if label_set else 1.0
            sample_weights.append(w)

        return torch.tensor(sample_weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────
def get_dataloaders(
    processed_dir: str,
    images_dir:    str,
    category_map:  Dict[str, int],
    batch_size:    int   = 64,
    img_size:      int   = 224,
    num_workers:   int   = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    ann_dir = Path(processed_dir) / "annotations"
    imgs    = Path(images_dir)

    print("\n[DataLoader] Building datasets …")
    train_ds = FashionMultiLabelDataset(
        str(ann_dir / "train.csv"), str(imgs), category_map,
        transform=get_train_transforms(img_size),
    )
    val_ds = FashionMultiLabelDataset(
        str(ann_dir / "val.csv"), str(imgs), category_map,
        transform=get_eval_transforms(img_size),
    )
    test_ds = FashionMultiLabelDataset(
        str(ann_dir / "test.csv"), str(imgs), category_map,
        transform=get_eval_transforms(img_size),
    )

    sampler = None
    if use_weighted_sampler:
        w       = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
