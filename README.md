# CV Hackathon Submission

This is Team 7-2-off and this is our submission to the Aaltoes Computer Vision Hackathon.
Team participants -> Christos Zonias, Iresh Gupta, and Oliver Raffone.

## Overview

In this project, we tackle the challenge of developing a model that detects AI-inpainted regions in images by generating binary segmentation masks, where each pixel is classified as either original (0) or manipulated (1), using pairs of original and altered training images. Our solution leverages modern computer vision techniques to preprocess the data, design and train a robust model, and evaluate its performance with clear visualizations and metrics.

## Key Features


Key Features
5-Fold Ensemble of EfficientNet-B7 U-Net models

Test-Time Augmentation (8 transformations per image)

Edge-Aware Post-Processing with Canny filtering

Custom Hybrid Loss: Dice + Focal + BCE

Mixed-Precision Training for faster convergence

## Approach

1. **Data Preparation**
Augmentation: Spatial/color transforms + CoarseDropout

Normalization: ImageNet stats

Dataset: 28,101 training pairs, 256x256 RGB

2. **Model Architecture**
python
Copy
smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
3. **Training Strategy**
5-Fold Cross-Validation

AdamW Optimizer (LR=1e-4, WD=1e-4)

Early Stopping (patience=7)

LR Scheduling on plateau

4. **Inference Pipeline**
Model ensembling across folds

TTA with rotations/flips

Canny edge refinement

RLE encoding for submission

A fully documented Jupyter Notebook (`cv_hackathon.ipynb`) walks through our entire workflow, including code explanations, parameter choices, and visualization of results.

## Repository Structure
```
cv_hackathon.ipynb 
.
├── train/                # Training data
│   ├── images/           # Manipulated images
│   └── masks/            # Ground truth masks
├── test/                 # Test data
│   └── images/           # Images to predict
├── main.py               # Full training/inference code
├── best_model_fold*.pth  # Saved model weights
└── submission.csv        # Final predictions 
README.md
LICENSE
requirements.txt       # (Optional) Dependencies file
```