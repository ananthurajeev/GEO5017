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
