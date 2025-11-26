import os
import cv2
import torch
from torchvision import models
import torch.nn as nn
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from configs import test_dir, IMG_SIZE, DEVICE
import matplotlib.pyplot as plt
import random

class TestModel:
    def __init__(self):
        self.class_names = sorted(os.listdir(test_dir))
        self.class_names = [c for c in self.class_names if os.path.isdir(os.path.join(test_dir, c))]
        self.model = self.load_model()
        self.test_tf = Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(),
            ToTensorV2(),
        ])
        
    def get_two_random_images_per_folder(self, root_dir):
        result = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            images = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if len(images) < 2: continue

            selected = random.sample(images, 2)
            result.extend(selected)

        return result

    
    def load_model(self, pth="best_model_pytorch.pth"):
        model = models.efficientnet_b0(weights=None)  
        model.classifier[1] = nn.Linear(1280, len(self.class_names))
        model.load_state_dict(torch.load(pth, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model

    def predict_image(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        transformed = self.test_tf(image=img_rgb)["image"]
        transformed = transformed.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(transformed)
            probs = torch.softmax(logits, dim=1)
            cid = torch.argmax(probs).item()
            conf = probs[0][cid].item()

        plt.imshow(img_rgb)
        plt.title(f"{self.class_names[cid]} ({conf*100:.2f}%)")
        plt.axis("off")
        plt.show()

        return self.class_names[cid], conf


    def test_image(self):
        selected_images = self.get_two_random_images_per_folder(test_dir)
        for fpath in selected_images:
            fname = os.path.basename(fpath)
            print("\nProcessing:", fname)
            self.predict_image(fpath)
