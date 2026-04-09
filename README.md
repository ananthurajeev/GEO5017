Urban Waste Detection Project
Overview

This project performs waste detection and classification using deep learning.
The main pipeline is implemented in a Jupyter Notebook and designed to run efficiently in Google Colab.

Project Structure
  Urban_Waste_Project.ipynb → Main pipeline (training, evaluation, prediction)
  GEO5017_Bonus_Task.ipynb → Standardisation and localisation (bonus tasks)


Packages required : 
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

Package installation:
pip install tensorflow numpy pandas matplotlib scikit-learn Pillow tqdm torch torchvision transformers opencv-python ultralytics supervision

For the main pipeline file 

Steps to be executed : 

  Mount Google Drive & install libraries
  Load & explore the CSV labels
  Build the image path mapping (handles subfolders)
  Train / Val / Test split (stratified)
  Dataset class + augmentations
  Train EfficientNet-B0 — 20 epochs, early stopping (patience=5), best model saved on waste F1
  Evaluate on test set — confusion matrix, ROC curve, Precision-Recall curve, threshold analysis
  Predict on unlabeled 4,000 images
  Export Top-100 detections


  For  standardisation and localisation:

  Prerequisite
Top-100 images must already exist in OUTPUT_DIR/top100_waste_detections/


