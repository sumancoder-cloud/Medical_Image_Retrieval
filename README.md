# Medical Image Retrieval Reproduction

This project reproduces the retrieval pipeline described in
`Medical_Image_Retrieval_System_for_Endoscopy_Images_Using_CNN-final (1).pdf`
using Kvasir-V2:

1. Use the eight Kvasir-V2 class directories as the dataset.
2. Select a deterministic test set of 200 images per class.
3. Use the remaining images for the feature database.
4. Preprocess images with ImageNet VGG-16 preprocessing.
5. Extract the 4096-dimensional output of VGG-16's first fully connected layer.
6. Rank database images by cosine similarity for every test query.
7. Report mean precision at 10, 20, and 30, per-class results, and recall.

The paper reports 872 training images and 200 test images per class, but
Kvasir-V2 contains only 1,000 images per class. Those two numbers cannot form
disjoint splits. The executed reproduction therefore uses 200 test images and
the remaining 800 images per class as the database. The scripts reject an
impossible request rather than overlapping test and database images.

## 1. Environment

From PowerShell:

```powershell
cd "C:\Users\user\OneDrive\ドキュメント\Desktop\UROP"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first feature-extraction run downloads the official ImageNet VGG-16
weights through torchvision. This requires internet access and several GB of
free disk space for the dataset, cached weights, and feature files.

## 2. Dataset

Download Kvasir-V2 from the official Simula dataset page:
<https://datasets.simula.no/kvasir-v2/>

Extract it so the source directory contains eight class directories. The
directory names may differ slightly in capitalization; the scripts normalize
them to the paper's labels:

```text
data\kvasir-v2\
  dyed-lifted-polyps\
  dyed-resection-margins\
  esophagitis\
  normal-cecum\
  normal-pylorus\
  normal-z-line\
  polyps\
  ulcerative-colitis\
```

Do not commit `data\`, `outputs\`, or model weights.

## 3. Reproduce the experiment

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py prepare_dataset.py `
  --source "data\kvasir-v2" `
  --output "outputs\paper_split" `
  --test-per-class 200 `
  --seed 2023

& $py extract_features.py `
  --manifest "outputs\paper_split\manifest.csv" `
  --output "outputs\paper_split\features.npz" `
  --batch-size 32

& $py evaluate_retrieval.py `
  --features "outputs\paper_split\features.npz" `
  --output-dir "outputs\paper_split\metrics"

& $py report_metrics.py `
  --metrics-dir "outputs\paper_split\metrics"
```

Or run all three stages with one command:

```powershell
.\run_reproduction.ps1 -Dataset "data\kvasir-v2"
```

The command above uses all 800 non-test images per class. Do not pass
`--train-per-class 872` for Kvasir-V2; that would require 1,072 images per
class.

## 4. Fine-tuned retrieval model

The baseline uses frozen VGG-16 ImageNet features. To train a stronger
embedding without touching the test queries, fine-tune ResNet-18 on the
`database` rows and export its penultimate-layer features:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py fine_tune_retrieval.py `
  --manifest "outputs\paper_split\manifest.csv" `
  --checkpoint "outputs\paper_split\resnet18_finetuned.pt" `
  --output-features "outputs\paper_split\resnet18_features.npz" `
  --epochs 12 `
  --batch-size 32 `
  --export-batch-size 128

& $py evaluate_retrieval.py `
  --features "outputs\paper_split\resnet18_features.npz" `
  --output-dir "outputs\paper_split\metrics_resnet18"
```

The script uses a stratified validation subset of database images, freezes the
early backbone, fine-tunes ResNet-18's final block and classifier, selects the
best validation checkpoint, and evaluates the untouched test split.
If training already produced the checkpoint, add `--resume` to skip training
and only export embeddings.

## 5. Web interface

Install the additional interface dependency and start the local app:

```powershell
& $py -m pip install -r requirements.txt
& $py -m streamlit run app.py
```

Open the displayed local URL, upload a JPG/PNG image, and choose any result
count from 10 through 30. The app uses the fine-tuned checkpoint and displays
the retrieved database images with their predicted class and cosine similarity.

## 6. Output files

| File | Meaning |
|---|---|
| `outputs/paper_split/manifest.csv` | Frozen train/test assignment and normalized class labels |
| `outputs/paper_split/features.npz` | VGG-16 FC1 embeddings and metadata |
| `outputs/paper_split/metrics/summary.json` | Overall precision@10/20/30 and recall |
| `outputs/paper_split/metrics/per_class.csv` | Precision@10/20/30 and recall by class |
| `outputs/paper_split/metrics/query_results.csv` | Every query's ranked retrieval metrics |
| `outputs/paper_split/metrics/top_retrievals.csv` | Top-30 retrieved paths and similarity scores |
| `outputs/paper_split/metrics/report.txt` | Human-readable overall and per-class report |

`summary.json` also records the exact split, model, preprocessing, and software
settings needed to audit a run. Results should not be claimed to match the
paper unless the dataset version, split, preprocessing, and model weights are
the same.

## 7. Reproducibility notes

- The split is deterministic and uses a stable hash-based ordering of paths.
- Feature extraction uses `torch.inference_mode()` and a fixed seed.
- Similarity uses L2-normalized embeddings and a dot product, which is cosine
  similarity.
- A query is relevant when the retrieved image has the same normalized class.
- Precision@K is relevant retrieved images divided by K.
- Recall is relevant retrieved images divided by the number of relevant images
  in the feature database for that query.
- The paper's tables use averaged class precision; this project reports both
  macro per-class averages and query-weighted averages.

The paper's published numbers may not be exactly reproducible because it does
not provide the original file split, augmentation seed, or feature files.
