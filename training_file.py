# Inpainting Detection Model for AaltoES 2025 Computer Vision Hackathon
# This code trains a segmentation model to detect inpainted regions in images
# and creates a submission file with RLE-encoded predictions

import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
from sklearn.model_selection import KFold

# Set seed for reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

# Define directories
TRAIN_IMG_DIR = "./train/train/images"
TRAIN_MASK_DIR = "./train/train/masks"
TEST_IMG_DIR = "./test/test/images"

# Parameters
IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 50
NUM_FOLDS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentations
def get_transforms(phase):
    if phase == "train":
        return A.Compose([
            A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(),
            ToTensorV2()
        ])
    else:  # val or test
        return A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(),
            ToTensorV2()
        ])

# Dataset class
class InpaintDataset(Dataset):
    def __init__(self, image_paths, mask_dir=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = os.path.basename(img_path).replace(".png", "")
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            # For test set, only return the image
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image, img_id
        else:
            # For train/val, return image and mask
            mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype("float32")
            
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]  # Tensor (3, 256, 256)
                mask = transformed["mask"].unsqueeze(0)  # Tensor (1, 256, 256)
            
            return image, mask

# Loss function
class DiceFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        # Focal Loss component
        focal = -self.alpha * (1 - probs)**self.gamma * targets * torch.log(probs + 1e-7)
        focal -= (1 - self.alpha) * probs**self.gamma * (1 - targets) * torch.log(1 - probs + 1e-7)
        focal = focal.mean()
        
        # Dice Loss component
        intersection = (probs * targets).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = 1 - (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice = dice.mean()
        
        return self.bce_weight * bce + self.focal_weight * focal + self.dice_weight * dice

# Metrics
def dice_metric(probs, targets):
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean(dim=1)

# Model initialization - FIXED
def initialize_model():
    model = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        # Remove attention_type parameter as it's not supported
    )
    return model.to(DEVICE)

# TTA (Test Time Augmentation)
def tta_predict(model, x):
    preds = []
    # Original + rotations
    for angle in [0, 90, 180, 270]:
        rotated = torch.rot90(x, k=angle//90, dims=[2,3])
        with torch.no_grad():
            pred = model(rotated)
        pred = torch.rot90(pred, k=-angle//90, dims=[2,3])
        preds.append(pred)
    
    # Flips
    for dim in [2, 3]:
        flipped = torch.flip(x, dims=[dim])
        with torch.no_grad():
            pred = model(flipped)
        pred = torch.flip(pred, dims=[dim])
        preds.append(pred)
    
    return torch.mean(torch.stack(preds), dim=0)

# Post-processing
def apply_canny(mask_np):
    mask_np = (mask_np * 255).astype(np.uint8)
    # Auto-threshold using Otsu's method
    high_thresh, _ = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(mask_np, low_thresh, high_thresh)
    # Combine edges with mask
    combined = np.clip(mask_np + edges, 0, 255)
    return (combined > 127).astype(np.uint8)

# RLE encoding
def rle_encode(mask):
    pixels = mask.flatten(order='C')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(r) for r in runs)

# Training function
def train_model(model, train_loader, val_loader, fold):
    criterion = DiceFocalBCELoss(bce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()
    
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 7  # Early stopping patience
    
    print(f"\n{'='*20} Training Fold {fold+1}/{NUM_FOLDS} {'='*20}")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        
        for images, masks in progress_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix(loss=loss.item())
        
        train_loss /= train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_total = 0.0
        val_samples = 0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Valid]")
        
        with torch.no_grad():
            for images, masks in progress_bar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                logits = model(images)
                loss = criterion(logits, masks)
                
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                dice = dice_metric(probs, masks)
                val_dice_total += dice.sum().item()
                val_samples += images.size(0)
                
                progress_bar.set_postfix(dice=dice.mean().item())
        
        val_loss /= len(val_loader)
        val_dice = val_dice_total / val_samples
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Dice={val_dice:.4f}")
        
        scheduler.step(val_dice)  # Adjust learning rate
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
            print(f"✓ Saved new best model (Val Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with Dice {best_dice:.4f}")
            break
    
    return best_dice

# Create K-fold datasets
def create_folds():
    all_img_paths = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
    print(f"Total training images: {len(all_img_paths)}")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    return list(kf.split(all_img_paths)), all_img_paths

# Main training loop
def train_kfold_models():
    folds, all_img_paths = create_folds()
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        train_img_paths = [all_img_paths[i] for i in train_idx]
        val_img_paths = [all_img_paths[i] for i in val_idx]
        
        # Create datasets
        train_dataset = InpaintDataset(
            train_img_paths, 
            mask_dir=TRAIN_MASK_DIR, 
            transform=get_transforms("train")
        )
        val_dataset = InpaintDataset(
            val_img_paths, 
            mask_dir=TRAIN_MASK_DIR, 
            transform=get_transforms("val")
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        model = initialize_model()
        best_dice = train_model(model, train_loader, val_loader, fold)
        fold_scores.append(best_dice)
        
        print(f"Fold {fold+1} Best Dice: {best_dice:.4f}")
        
        # Clear memory
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
    
    # Print fold results
    print("\n" + "="*50)
    print("K-fold Cross-validation Results:")
    for fold, score in enumerate(fold_scores):
        print(f"Fold {fold+1}: {score:.4f}")
    print(f"Mean Dice: {np.mean(fold_scores):.4f}")
    print("="*50)

# Generate predictions for test set
def predict_test():
    # Load test images
    test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    print(f"Total test images: {len(test_img_paths)}")
    
    # Create test dataset
    test_dataset = InpaintDataset(
        test_img_paths,
        transform=get_transforms("test"),
        is_test=True
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Load all fold models
    models = []
    for fold in range(NUM_FOLDS):
        model_path = f"best_model_fold{fold}.pth"
        if os.path.exists(model_path):
            model = initialize_model()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models.append(model)
            print(f"Loaded model from {model_path}")
    
    if not models:
        raise ValueError("No model weights found. Please train models first.")
    
    # Generate predictions
    results = []
    progress_bar = tqdm(test_loader, desc="Generating predictions")
    
    for images, img_ids in progress_bar:
        images = images.to(DEVICE)
        
        # Ensemble predictions from all models
        ensemble_preds = []
        for model in models:
            with torch.no_grad():
                pred = torch.sigmoid(tta_predict(model, images))
                ensemble_preds.append(pred)
        
        # Average predictions
        final_preds = torch.mean(torch.stack(ensemble_preds), dim=0)
        
        # Process each prediction
        for i, img_id in enumerate(img_ids):
            pred = final_preds[i].squeeze().cpu().numpy()
            
            # Apply threshold
            pred_bin = (pred > 0.5).astype(np.uint8)
            
            # Apply post-processing
            pred_canny = apply_canny(pred_bin)
            
            # RLE encode
            rle = rle_encode(pred_canny)
            results.append([img_id, rle])
    
    # Create submission file
    submission_df = pd.DataFrame(results, columns=["ImageId", "EncodedPixels"])
    submission_df.to_csv("submission(1).csv", index=False)
    print(f"Submission file created with {len(submission_df)} entries.")
    
    return submission_df

if __name__ == "__main__":
    # Train models
    train_kfold_models()
    
    # Generate predictions and submission file
    submission_df = predict_test()
    
    print("Done! The submission file has been created.")