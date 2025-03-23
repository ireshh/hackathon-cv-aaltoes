# CV Hackathon Submission

This is our submission to the Aaltoes Computer Vision Hackathon. Our team is 7-2-Off Team participants include Christos Zonias, Iresh Gupta, and Oliver Raffone.

## Overview

In this project, we tackle the challenge of developing a model that detects AI-inpainted regions in images by generating binary segmentation masks, where each pixel is classified as either original (0) or manipulated (1), using pairs of original and altered training images. Our solution leverages modern computer vision techniques to preprocess the data, design and train a robust model, and evaluate its performance with clear visualizations and metrics.

## Approach

Our solution follows these key steps:

1. **Data Preprocessing:**
   - Data cleaning, augmentation, and normalization to prepare the dataset.
2. **Model Design:**
   - Implementation of a model architecture based on [explain architecture, e.g., Convolutional Neural Networks, transfer learning, etc.].
3. **Training:**
   - Hyperparameter tuning, training with validation, and model optimization.
4. **Evaluation:**
   - Assessment using metrics such as accuracy, precision, recall, and F1 score, along with visualizations for qualitative analysis.

A fully documented Jupyter Notebook (`cv_hackathon.ipynb`) walks through our entire workflow, including code explanations, parameter choices, and visualization of results.

## Repository Structure
```
cv_hackathon.ipynb 
/models
    ├── model.py         # Contains model definitions, training scripts, or saved weights  
    └── additional_files # (e.g., pretrained weights, helper scripts, etc.)    
README.md
LICENSE
requirements.txt       # (Optional) Dependencies file
```