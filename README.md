# GEO5017 — Urban Waste Detection: ConvNeXt-Base

Binary waste detection in street-level imagery using **ConvNeXt-Base** pretrained on ImageNet-22k.

---

## Repository Structure

```
GEO5017-UrbanWaste-ConvNeXt/
├── GEO5017_ConvNeXt_Base.ipynb   # Main notebook (run in Google Colab)
├── README.md
└── outputs/                      # Generated after running the notebook
    ├── best_convnext.pt           # Best model checkpoint
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

## Data Setup (Google Drive)

Place your files in Google Drive exactly as shown:

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

---

## Data Splits

| Split | Years | Role |
|---|---|---|
| Train | 2016–2019 | Model training |
| Val | 2020–2021 | Hyperparameter tuning & early stopping |
| Test | 2022–2023 | Final held-out evaluation |

---

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | `convnext_base.fb_in22k_ft_in1k` via `timm` |
| Pretraining | ImageNet-22k → fine-tuned on ImageNet-1k |
| Feature dim | 1024 |
| Head | LayerNorm → Dropout(0.3) → Linear(1024→256) → GELU → Dropout(0.3) → Linear(256→2) |
| Parameters | ~87M (fully fine-tuned) |

---

## Training Details

| Setting | Value |
|---|---|
| Loss | Focal Loss (γ=2, class-weighted) |
| Sampler | WeightedRandomSampler — ~50% waste per batch |
| Backbone LR | 1e-4 |
| Head LR | 4e-4 |
| Weight decay | 1e-4 |
| LR schedule | Linear warm-up (3 epochs) → cosine decay |
| Max epochs | 30 |
| Early stop | Patience 7 on val F1 (waste) |
| Batch size | 32 |
| Image size | 224×224 |

---

## Running in Google Colab

1. Upload `GEO5017_ConvNeXt_Base.ipynb` to Google Colab (or open from Drive).
2. Set **Runtime → Change runtime type → GPU**.
3. Run cells top-to-bottom (Cell 1 → Cell 15).
4. Outputs are saved to `MyDrive/Waste/`.

---

## Requirements

Installed automatically in Cell 1:

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

## Notebook Cells

| # | Cell | Description |
|---|---|---|
| 1 | Setup | Install packages, imports, random seeds |
| 2 | Mount Drive & Paths | Mount Google Drive, define all paths |
| 3 | Load Labels | Read `waste.csv`, map labels, assign year-based splits |
| 4 | Dataset & Sampler | Augmentation pipelines, `WasteDataset`, `WeightedRandomSampler` |
| 5 | ConvNeXt-Base Model | Model definition with custom classification head |
| 6 | Training | Focal Loss, AdamW, LR schedule, early stopping |
| 7 | Training Curves | Loss, F1/accuracy, and LR plots |
| 8 | Threshold Optimisation | Sweep decision threshold on val set; PR curve |
| 9 | Validation Report | Classification report, confusion matrix, P@100 |
| 10 | Test Evaluation | Full test set metrics, confusion matrix, score histogram |
| 11 | Top-100 Breakdown | Bar chart of top-100 ranked predictions |
| 12 | Submission | Copy top-100 images + export `top100_ranked.csv` |
| 13 | Top-20 Visualisation | Grid of top-20 detections with GT labels |
| 14 | Error Analysis | False positive and false negative visualisations |
| 15 | Results Table | Final metric summary + heatmap |
