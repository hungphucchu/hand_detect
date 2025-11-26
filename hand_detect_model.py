import os
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from configs import DATASET_DIR, BATCH_SIZE, EPOCHS, DEVICE, train_tf, val_tf
from gesture_dataset import GestureDataset

class HandDetectModel:
    def __init__(self):
        self.TRAIN_SUBJ = ["00","01","02","03","04","05","06"]
        self.VAL_SUBJ   = ["07"]
        self.TEST_SUBJ  = ["08","09"]
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def collect_files(self, subjects):
        files, labels = [], []
        classes = set()

        for subj in subjects:
            subj_dir = os.path.join(DATASET_DIR, subj)
            for gesture in os.listdir(subj_dir):
                gpath = os.path.join(subj_dir, gesture)
                if not os.path.isdir(gpath):
                    continue

                classes.add(gesture)
                for img in os.listdir(gpath):
                    if img.endswith((".png",".jpg",".jpeg")):
                        files.append(os.path.join(gpath, img))
                        labels.append(gesture)

        return files, labels, sorted(list(classes))
    
    def prepareData(self):
        train_files, train_labels, class_names = self.collect_files(self.TRAIN_SUBJ)
        val_files,   val_labels,   _           = self.collect_files(self.VAL_SUBJ)
        test_files,  test_labels,  _           = self.collect_files(self.TEST_SUBJ)
        self.model.classifier[1] = nn.Linear(1280, len(class_names))
        self.model = self.model.to(DEVICE)

        class_to_idx = {c:i for i,c in enumerate(class_names)}

        train_labels = [class_to_idx[y] for y in train_labels]
        val_labels   = [class_to_idx[y] for y in val_labels]
        test_labels  = [class_to_idx[y] for y in test_labels]


        train_ds = GestureDataset(train_files, train_labels, train_tf)
        val_ds   = GestureDataset(val_files,   val_labels,   val_tf)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
        return train_loader, val_loader

    def training(self):
        best_val_acc = 0
        train_loader, val_loader = self.prepareData()

        for epoch in range(EPOCHS):
            self.model.train()
            total, correct = 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                preds = self.model(imgs)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = preds.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix(acc=correct/total)

            self.model.eval()
            val_total, val_correct = 0, 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds = self.model(imgs)
                    _, predicted = preds.max(1)

                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total
            print(f"VAL ACC: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_model_pytorch.pth")
                print("Saved best_model_pytorch.pth")

        print("Training complete!")
        print("Best validation accuracy:", best_val_acc)
