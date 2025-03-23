import os, cv2, glob, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_transforms(phase):
    if phase == "train":
        return A.Compose([
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2)),
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
    else:
        return A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(),
            ToTensorV2()
        ])

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
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image, img_id
        else:
            mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype("float32")
            
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"].unsqueeze(0)
            
            return image, mask

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
        focal = -self.alpha * (1 - probs)**self.gamma * targets * torch.log(probs + 1e-7)
        focal -= (1 - self.alpha) * probs**self.gamma * (1 - targets) * torch.log(1 - probs + 1e-7)
        focal = focal.mean()
        
        intersection = (probs * targets).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = 1 - (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice = dice.mean()
        
        return self.bce_weight * bce + self.focal_weight * focal + self.dice_weight * dice

def dice_metric(probs, targets):
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean(dim=1)

def initialize_model():
    model = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

def tta_predict(model, x):
    preds = []
    for angle in [0, 90, 180, 270]:
        rotated = torch.rot90(x, k=angle//90, dims=[2,3])
        with torch.no_grad():
            pred = model(rotated)
        pred = torch.rot90(pred, k=-angle//90, dims=[2,3])
        preds.append(pred)
    
    for dim in [2, 3]:
        flipped = torch.flip(x, dims=[dim])
        with torch.no_grad():
            pred = model(flipped)
        pred = torch.flip(pred, dims=[dim])
        preds.append(pred)
    
    return torch.mean(torch.stack(preds), dim=0)

def apply_canny(mask_np):
    mask_np = (mask_np * 255).astype(np.uint8)
    high_thresh, _ = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(mask_np, low_thresh, high_thresh)
    combined = np.clip(mask_np + edges, 0, 255)
    return (combined > 127).astype(np.uint8)

def rle_encode(mask):
    pixels = mask.flatten(order='C')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(r) for r in runs)

def create_folds(train_img_dir):
    all_img_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return list(kf.split(all_img_paths)), all_img_paths

def hello_world():
    print("Hello from utils!")