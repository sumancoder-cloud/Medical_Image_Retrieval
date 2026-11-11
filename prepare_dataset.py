"""Create a deterministic Kvasir-V2 train/database and test manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path



CLASS_ALIASES = {
    "dyed-lifted-polyps": "DLP",
    "dyed-resection-margins": "DRM",
    "esophagitis": "ESO",
    "normal-cecum": "NC",
    "normal-pylorus": "NP",
    "normal-z-line": "NZL",
    "polyps": "POLY",
    "ulcerative-colitis": "ULCO",
}



def stable_key(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path.as_posix()}".encode("utf-8")).hexdigest()


def find_class_directories(source: Path) -> dict[str, Path]:
    directories = {path.name.lower().replace("_", "-"): path for path in source.iterdir() if path.is_dir()}
    missing = sorted(set(CLASS_ALIASES) - set(directories))
    if missing:
        raise FileNotFoundError(
            "Missing Kvasir-V2 class directories: "
            + ", ".join(missing)
            + f". Found: {', '.join(sorted(directories)) or '(none)'}"
        )
    return {CLASS_ALIASES[name]: directories[name] for name in CLASS_ALIASES}


def build_manifest(source: Path, test_per_class: int, train_per_class: int | None, seed: int) -> list[dict[str, str]]:
    class_directories = find_class_directories(source)
    rows: list[dict[str, str]] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label, directory in class_directories.items():
        images = sorted(
            (path for path in directory.rglob("*") if path.suffix.lower() in image_extensions),
            key=lambda path: stable_key(path.relative_to(source), seed),
        )
        if len(images) < test_per_class + (train_per_class or 0):
            raise ValueError(
                f"{label} has {len(images)} images, but "
                f"{test_per_class + (train_per_class or 0)} are required."
            )
        test_images = images[:test_per_class]
        remaining = images[test_per_class:]
        database_images = remaining[:train_per_class] if train_per_class is not None else remaining
        for split, selected in (("test", test_images), ("database", database_images)):
            rows.extend(
                {"path": str(path.resolve()), "class": label, "split": split}
                for path in selected
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-per-class", type=int, default=200)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    if args.test_per_class <= 0 or args.train_per_class is not None and args.train_per_class <= 0:
        raise ValueError("Image counts must be positive.")
    rows = build_manifest(args.source.resolve(), args.test_per_class, args.train_per_class, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
