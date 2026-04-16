# Urban Waste Detection — ConvNeXt-Base

## Project Overview

This project performs **binary waste detection in street-level urban imagery using deep learning**.  
The pipeline is implemented in a Jupyter Notebook and optimised for execution in **Google Colab**.

The model uses **ConvNeXt-Base** pretrained on ImageNet-22k (via `timm`), fine-tuned end-to-end  
on the UrbanWaste dataset with Focal Loss and a balanced sampling strategy to handle class imbalance.

---

## Project Structure

```
GEO5017_ConvNeXt_Base.ipynb    # Main pipeline (training, evaluation, top-100 prediction)
README.md
.gitignore
```

---

## Required Packages

```
timm
torch
torchvision
tqdm
scikit-learn
matplotlib
seaborn
Pillow
```

---

## Installation

Install all required packages using:

```bash
pip install timm torch torchvision tqdm scikit-learn matplotlib seaborn Pillow
```

> All packages are also installed automatically in **Cell 1** of the notebook.

---

## Data Setup

Place your files in Google Drive exactly as shown below before running the notebook:

```
MyDrive/
└── Waste/
    ├── waste.csv
    └── UrbanWaste-images-10k-right/
        ├── year_2016/
        ├── year_2017/
        ├── year_2018/
        ├── year_2019/
        ├── year_2020/
        ├── year_2021/
        ├── year_2022/
        └── year_2023/
```

Update these paths in **Cell 2** if your Drive folder structure is different:

```python
BASE_DIR    = '/content/drive/MyDrive/Waste/UrbanWaste-images-10k-right'
LABELS_FILE = '/content/drive/MyDrive/Waste/waste.csv'
OUTPUT_DIR  = '/content/drive/MyDrive/Waste'
```

---

## How to Run

### Run in Google Colab (Recommended)

1. Upload `GEO5017_ConvNeXt_Base.ipynb` to Google Colab
2. Set **Runtime → Change runtime type → GPU** (T4 or better)
3. Mount your Google Drive when prompted in Cell 2
4. Run all cells top-to-bottom (Cell 1 → Cell 15)

---

## Execution Steps

### Cell 1 — Setup
- Install required packages
- Import all libraries
- Set random seeds for reproducibility
- Detect and print GPU device

### Cell 2 — Mount Drive & Set Paths
- Mount Google Drive
- Define all file paths and output directory

### Cell 3 — Load Labels & Define Splits
- Read `waste.csv` and rename columns
- Map labels: `waste → 1`, `no_waste → 0`
- Assign year-based splits:

| Split | Years | Purpose |
|---|---|---|
| Train | 2016–2019 | Model training |
| Val | 2020–2021 | Hyperparameter tuning & early stopping |
| Test | 2022–2023 | Final held-out evaluation |

### Cell 4 — Dataset, Augmentation & Balanced Sampler
- Define training augmentations:
  - Random horizontal/vertical flip
  - Colour jitter (brightness, contrast, saturation, hue)
  - Random rotation (±15°)
  - Random grayscale
  - Random erasing (occlusion simulation)
- Define `WasteDataset` class
- Create `WeightedRandomSampler` to balance waste/clean batches (~50/50)
- Build `DataLoader` objects for train, val, and test sets

### Cell 5 — ConvNeXt-Base Model
- Load `convnext_base.fb_in22k_ft_in1k` backbone via `timm` (pretrained on ImageNet-22k)
- Attach custom classification head:
  - `LayerNorm → Dropout → Linear(1024→256) → GELU → Dropout → Linear(256→2)`
- Full fine-tuning (~87M trainable parameters)

### Cell 6 — Training
- **Loss:** Focal Loss (γ=2) with class weights
- **Optimiser:** AdamW with differential learning rates:
  - Backbone: `1e-4`
  - Head: `4e-4`
- **LR Schedule:** Linear warm-up (3 epochs) → cosine decay
- **Max epochs:** 30 with early stopping (patience = 7 on val F1)
- Best model saved as `best_convnext.pt`

### Cell 7 — Training Curves
- Focal loss curve (train vs val)
- Val F1 and accuracy over epochs
- Learning rate schedule plot
- Saved as `training_curves.png`

### Cell 8 — Threshold Optimisation
- Load best checkpoint
- Run inference on validation set
- Sweep decision threshold (0.05–0.95) to maximise F1 on waste class
- Plot Precision-Recall curve (AP score)
- Saved as `threshold_tuning.png`

### Cell 9 — Validation Report
- Classification report (precision, recall, F1 per class)
- Confusion matrix at optimal threshold
- ROC-AUC and Average Precision
- Precision@100 on validation set
- Saved as `confusion_val.png`

### Cell 10 — Test Set Evaluation
- Full test set inference with best checkpoint
- Classification report, confusion matrix, score distribution histogram
- Reports: Accuracy, Precision, Recall, F1, ROC-AUC, Avg Precision, P@100
- Saved as `test_evaluation.png`

### Cell 11 — Top-100 Breakdown
- Rank all test images by predicted waste probability
- Bar chart of top-100 scores (red = true positive, blue = false positive)
- Saved as `top100_breakdown.png`

### Cell 12 — Generate Top-100 Submission
- Copy top-100 ranked images to `top100_ConvNeXt_Base_submission/`
- File names encode rank, score, and ground-truth label
- Export `top100_ranked.csv` with full metadata

### Cell 13 — Visualise Top-20 Detections
- 4×5 grid of the top-20 ranked images
- Green title = correct waste detection, red = false positive
- Saved as `top20_detections.png`

### Cell 14 — Error Analysis
- Top-8 false positives (highest-scoring clean images predicted as waste)
- Top-8 false negatives (highest-confidence missed waste images)
- Saved as `errors_fp.png` and `errors_fn.png`

### Cell 15 — Final Results Table
- Summary table of val and test metrics
- Heatmap visualisation
- Saved as `results.csv` and `results_heatmap.png`

---

## Outputs

All outputs are saved to `OUTPUT_DIR` (default: `MyDrive/Waste/`):

```
Waste/
├── best_convnext.pt
├── training_curves.png
├── threshold_tuning.png
├── confusion_val.png
├── test_evaluation.png
├── top100_breakdown.png
├── top20_detections.png
├── errors_fp.png
├── errors_fn.png
├── results.csv
├── results_heatmap.png
└── top100_ConvNeXt_Base_submission/
    ├── top100_ranked.csv
    └── rank001_score0.xxx_GT1_<filename>.jpg  ...
```

---

## Model Summary

| Setting | Value |
|---|---|
| Backbone | `convnext_base.fb_in22k_ft_in1k` (timm) |
| Pretraining | ImageNet-22k → fine-tuned on ImageNet-1k |
| Feature dimension | 1024 |
| Parameters | ~87M (fully fine-tuned) |
| Loss | Focal Loss (γ=2, class-weighted) |
| Sampler | WeightedRandomSampler (~50% waste per batch) |
| Backbone LR | 1e-4 |
| Head LR | 4e-4 |
| LR schedule | Warm-up (3 ep) → cosine decay |
| Max epochs | 30 |
| Early stopping | Patience 7 on val F1 (waste) |
| Batch size | 32 |
| Image size | 224 × 224 |
