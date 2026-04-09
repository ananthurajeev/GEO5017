# Urban Waste Detection Project

## Project Overview
This project performs **waste detection and classification using deep learning**.  
The main pipeline is implemented in a Jupyter Notebook and optimized for execution in **Google Colab**.

---

## Project Structure
```
Urban_Waste_Project.ipynb      # Main pipeline (training, evaluation, prediction)
GEO5017_Bonus_Task.ipynb      # Standardisation and localisation (bonus tasks)
```

---

## Required Packages
```
tensorflow>=2.12
numpy
pandas
matplotlib
scikit-learn
Pillow
tqdm
torch
torchvision
transformers
opencv-python
ultralytics
supervision
```

---

## Installation

Install all required packages using:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn Pillow tqdm torch torchvision transformers opencv-python ultralytics supervision
```

---

## How to Run

### Run in Google Colab (Recommended)

1. Upload the notebooks to Google Colab  
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Set dataset paths, for example:
   ```python
   CSV_PATH = '/content/drive/MyDrive/GEO5017/labels.csv'
   IMAGES_ROOT = '/content/drive/MyDrive/GEO5017/UrbanWaste-images-10k-right'
   ```

---

### Step 1 — Main Pipeline

Run:
```
Urban_Waste_Project.ipynb
```

#### Execution Steps
- Mount Google Drive and install libraries  
- Load and explore the CSV labels: label.csv attached in the zip folder
- Build image path mapping (handles subfolders)  
- Perform stratified Train / Validation / Test split  
- Create dataset class with augmentations  
- Train EfficientNet-B0:
  - 20 epochs  
  - Early stopping (patience = 5)  
  - Best model saved based on waste F1 score  
- Evaluate on test set:
  - Confusion matrix  
  - ROC curve  
  - Precision-Recall curve  
  - Threshold analysis  
- Predict on 4,000 unlabeled images  
- Export Top-100 detections  

---

### Step 2 — Standardisation & Localisation (Bonus)

Run:
```
GEO5017_Bonus_Task.ipynb
```

#### Prerequisite
Before running the bonus notebook, ensure:

```
OUTPUT_DIR/top100_waste_detections/
```

This directory must contain the **Top-100 detected images** generated from the main pipeline.

---

## Notes
- The project was developed and tested in **Google Colab** for computational efficiency  
- Ensure Google Drive paths are correctly set before running  
- GPU runtime is recommended for faster training  

---

## Future Improvements (Optional)
- Compare EfficientNet variants (B0 vs B3)  
- Integrate YOLO-based detection into main pipeline  
- Improve class imbalance handling  
- Add real-time inference support  
