import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import wandb
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassF1Score, MulticlassConfusionMatrix
)

# === Initialize Weights & Biases (W&B) for tracking ===
wandb.init(
    project="Diabetes Retinopathy",
    name="vit-base-adamw-metrics-optimized",
    config={
        "model": "ViT-Base (Patch 16)",
        "optimizer": "AdamW",
        "epochs": 30,
        "learning_rate": 5e-5,
        "batch_size": 8
    }
)

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define batch size
BATCH_SIZE = 8  

# ✅ Dataset Paths
dataset_path = "datasets/train_test_val"
data_dirs = {x: os.path.join(dataset_path, x) for x in ["train", "val"]}

# ✅ Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Load datasets
image_datasets = {x: datasets.ImageFolder(data_dirs[x], transform=transform) for x in ["train", "val"]}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) for x in ["train", "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
num_classes = len(class_names)

# ✅ Compute Dynamic Class Weights
labels_list = [label for _, label in image_datasets["train"].imgs]  
class_weights_auto = compute_class_weight(class_weight="balanced", classes=np.unique(labels_list), y=labels_list)
class_weights_auto = torch.tensor(class_weights_auto, dtype=torch.float).to(device)
print(f"Computed Class Weights: {class_weights_auto}")

# ✅ Load Pretrained ViT Model
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
model = model.to(device)

# ✅ Define Focal Loss Function
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, label_smoothing=0.1)(inputs, targets)  # Label Smoothing Added
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=class_weights_auto, gamma=2)  # Using computed class weights

# ✅ Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# ✅ Mixed Precision Training
scaler = torch.amp.GradScaler()

# ✅ Early Stopping Configuration
early_stopping_patience = 5
best_loss = float("inf")
epochs_no_improve = 0

# ✅ Torchmetrics Initialization (Restored)
train_accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
val_accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
train_precision = MulticlassPrecision(num_classes=num_classes).to(device)
val_precision = MulticlassPrecision(num_classes=num_classes).to(device)
train_recall = MulticlassRecall(num_classes=num_classes).to(device)
val_recall = MulticlassRecall(num_classes=num_classes).to(device)
train_f1 = MulticlassF1Score(num_classes=num_classes).to(device)
val_f1 = MulticlassF1Score(num_classes=num_classes).to(device)
confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true').to(device)

# ✅ Mixup Data Augmentation Function
def mixup_data(x, y, alpha=0.2):
    """Apply Mixup regularization"""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ✅ Training Function (Restored Metrics)
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=30):
    global best_loss, epochs_no_improve

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        wandb.log({"epoch": epoch + 1})

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # Apply Mixup in Training Phase
                if phase == "train":
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)

                    # Apply Mixup loss
                    if phase == "train":
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
                        loss = criterion(outputs, labels)

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds)
                all_labels.append(labels)

            # Compute Metrics
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            acc = train_accuracy(all_preds, all_labels) if phase == "train" else val_accuracy(all_preds, all_labels)
            precision = train_precision(all_preds, all_labels) if phase == "train" else val_precision(all_preds, all_labels)
            recall = train_recall(all_preds, all_labels) if phase == "train" else val_recall(all_preds, all_labels)
            f1 = train_f1(all_preds, all_labels) if phase == "train" else val_f1(all_preds, all_labels)
            cm = confusion_matrix(all_preds, all_labels)

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            wandb.log({
                f"{phase}_loss": epoch_loss,
                f"{phase}_accuracy": acc,
                f"{phase}_precision": precision,
                f"{phase}_recall": recall,
                f"{phase}_f1": f1
            }, commit=True)

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), "best_vit_model.pth")
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered.")
                    return model

        torch.cuda.empty_cache()
        gc.collect()

    return model

# ✅ Start Training
model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=30)
torch.save(model.state_dict(), "vit_final_model.pth")
wandb.finish()