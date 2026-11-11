"""Extract VGG-16 FC1 embeddings for a manifest."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[dict[str, str]], class_to_id: dict[str, int], transform) -> None:
        self.rows = rows
        self.class_to_id = class_to_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_id[row["class"]]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.manifest)
    if not rows:
        raise ValueError("Manifest is empty.")
    labels = sorted({row["class"] for row in rows})
    class_to_id = {label: index for index, label in enumerate(labels)}
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights).eval()
    feature_model = torch.nn.Sequential(*list(model.classifier.children())[:2]).eval()
    transform = weights.transforms()
    dataset = ManifestDataset(rows, class_to_id, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.features = model.features.to(device)
    feature_model = feature_model.to(device)
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, label_ids in loader:
            features = feature_model(model.features(images.to(device)).flatten(1))
            all_features.append(features.cpu().numpy().astype(np.float32))
            all_labels.append(label_ids.numpy().astype(np.int64))
    features_array = np.concatenate(all_features)
    labels_array = np.concatenate(all_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features_array,
        labels=labels_array,
        paths=np.asarray([row["path"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        classes=np.asarray(labels),
        model=np.asarray(["vgg16_imagenet1k_v1_classifier_fc1"]),
        input_size=np.asarray([224, 224]),
    )
    print(f"Wrote {len(rows)} embeddings of shape {features_array.shape} to {args.output}")


if __name__ == "__main__":
    main()
