import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

basePath = os.path.dirname(os.path.abspath(__file__))

BATCH_SIZE = 16
EPOCHS = 20
IMG_SIZE = 224
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATASET_DIR = os.path.join(basePath, "leapGestRecog")
test_dir   = os.path.join(DATASET_DIR, "09")

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Normalize(),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])