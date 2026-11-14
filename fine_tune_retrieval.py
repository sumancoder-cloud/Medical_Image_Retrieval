"""Fine-tune an ImageNet backbone and export embeddings for retrieval."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop
from torchvision.transforms import CenterCrop, Resize, ToTensor
from torchvision.transforms import ColorJitter, Normalize


class ImageRows(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[dict[str, str]], labels: dict[str, int], transform: Compose) -> None:
        self.rows = rows
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[row["class"]]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_training_rows(rows: list[dict[str, str]], seed: int, validation_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["class"], []).append(row)
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for class_name, class_rows in sorted(grouped.items()):
        shuffled = class_rows.copy()
        random.Random(f"{seed}:{class_name}").shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])
    return train, validation


def make_transforms(weights: ResNet18_Weights) -> tuple[Compose, Compose]:
    mean = weights.transforms().mean
    std = weights.transforms().std
    normalize = Normalize(mean, std)
    train_transform = Compose(
        [
            RandomResizedCrop(224, scale=(0.75, 1.0)),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = Compose([Resize(256), CenterCrop(224), ToTensor(), normalize])
    return train_transform, evaluation_transform


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.size(0)
    return total_loss / count, correct / count


def export_embeddings(
    model: nn.Module,
    rows: list[dict[str, str]],
    labels: dict[str, int],
    classes: list[str],
    transform: Compose,
    batch_size: int,
    device: torch.device,
    output: Path,
) -> None:
    dataset = ImageRows(rows, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_model = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    features: list[np.ndarray] = []
    label_values: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            batch_features = feature_model(images.to(device)).flatten(1)
            features.append(batch_features.cpu().numpy().astype(np.float32))
            label_values.append(batch_labels.numpy().astype(np.int64))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(features),
        labels=np.concatenate(label_values),
        paths=np.asarray([row["path"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        classes=np.asarray(classes),
        model=np.asarray(["resnet18_finetuned_layer4_avgpool"]),
        input_size=np.asarray([224, 224]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--export-batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    if args.epochs <= 0 or not 0 < args.validation_fraction < 1:
        raise ValueError("epochs must be positive and validation-fraction must be between 0 and 1.")
    if args.export_batch_size <= 0 or args.threads < 0:
        raise ValueError("export-batch-size must be positive and threads cannot be negative.")
    if args.threads:
        torch.set_num_threads(args.threads)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.manifest)
    classes = sorted({row["class"] for row in rows})
    labels = {class_name: index for index, class_name in enumerate(classes)}
    database_rows = [row for row in rows if row["split"] == "database"]
    train_rows, validation_rows = split_training_rows(database_rows, args.seed, args.validation_fraction)
    weights = ResNet18_Weights.IMAGENET1K_V1
    train_transform, evaluation_transform = make_transforms(weights)
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        ImageRows(train_rows, labels, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        ImageRows(validation_rows, labels, evaluation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    else:
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.layer4.parameters(), "lr": 1e-4},
                {"params": model.fc.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )
        best_accuracy = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(model, train_loader, loss_function, optimizer, device)
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, loss_function, None, device
            )
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
            )
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "classes": classes,
                        "validation_accuracy": best_accuracy,
                    },
                    args.checkpoint,
                )
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    export_embeddings(
        model,
        rows,
        labels,
        classes,
        evaluation_transform,
        args.export_batch_size,
        device,
        args.output_features,
    )
    print(
        f"Wrote fine-tuned embeddings to {args.output_features}; "
        f"best validation accuracy={checkpoint['validation_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
