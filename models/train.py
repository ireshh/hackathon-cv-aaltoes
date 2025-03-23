from utils.helpers import (
    seed_everything,
    get_transforms,
    InpaintDataset,
    initialize_model,
    DiceFocalBCELoss,
    dice_metric,
    tta_predict,
    apply_canny,
    rle_encode,
    create_folds
)
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Set device, seed and define directories
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

# Set seed and define directories
seed_everything(42)
TRAIN_IMG_DIR = "./data/train/images"
TRAIN_MASK_DIR = "./data/train/masks"
TEST_IMG_DIR = "./data/test/images"

NUM_FOLDS = 5
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

def train_folds(model, train_loader, val_loader, fold):
    criterion = DiceFocalBCELoss(bce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
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

def train_model_wrapper():
    folds, all_img_paths = create_folds(TRAIN_IMG_DIR)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        train_img_paths = [all_img_paths[i] for i in train_idx]
        val_img_paths = [all_img_paths[i] for i in val_idx]
        train_dataset = InpaintDataset(train_img_paths, mask_dir=TRAIN_MASK_DIR, transform=get_transforms("train"))
        val_dataset = InpaintDataset(val_img_paths, mask_dir=TRAIN_MASK_DIR, transform=get_transforms("val"))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        model = initialize_model()
        best_dice = train_folds(model, train_loader, val_loader, fold)
        fold_scores.append(best_dice)
        print(f"Fold {fold+1} Best Dice: {best_dice:.4f}")
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
    print("K-fold Cross-validation Results:")
    for fold, score in enumerate(fold_scores):
        print(f"Fold {fold+1}: {score:.4f}")

def predict_test():
    test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    print(f"Total test images: {len(test_img_paths)}")
    
    test_dataset = InpaintDataset(test_img_paths, transform=get_transforms("test"), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
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
    
    results = []
    progress_bar = tqdm(test_loader, desc="Generating predictions")
    
    for images, img_ids in progress_bar:
        images = images.to(DEVICE)
        
        ensemble_preds = []
        for model in models:
            with torch.no_grad():
                pred = torch.sigmoid(tta_predict(model, images))
                ensemble_preds.append(pred)
        
        final_preds = torch.mean(torch.stack(ensemble_preds), dim=0)
        
        for i, img_id in enumerate(img_ids):
            pred = final_preds[i].squeeze().cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)
            pred_canny = apply_canny(pred_bin)
            rle = rle_encode(pred_canny)
            results.append([img_id, rle])
    
    submission_df = pd.DataFrame(results, columns=["ImageId", "EncodedPixels"])
    submission_df.to_csv("submission.csv", index=False)
    print(f"Submission file created with {len(submission_df)} entries.")
    
    return submission_df

if __name__ == "__main__":
    train_model_wrapper()
    submission_df = predict_test()
    print("Done! The submission file has been created.")