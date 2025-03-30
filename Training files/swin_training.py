import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import wandb
import os
import gc
from sklearn.metrics import classification_report

# === Initialize Weights & Biases (W&B) ===
wandb.init(
    project="Diabetes Retinopathy",
    name="swin-base-20epochs-adamw-LRS-ES",
    config={
        "model": "Swin Transformer (Base)",
        "optimizer": "AdamW",
        "epochs": 20,
        "learning_rate": 1e-4,
        "batch_size": 8,
    }
)

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define batch size
BATCH_SIZE = 8  

# ✅ Dataset Paths
dataset_path = "datasets/train_test_val"
data_dirs = {x: os.path.join(dataset_path, x) for x in ["train", "val", "test"]}

# ✅ Data Preprocessing (No Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Swin Transformer expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
])

# ✅ Load datasets
image_datasets = {x: datasets.ImageFolder(data_dirs[x], transform=transform) for x in ["train", "val"]}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    for x in ["train", "val"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
num_classes = len(image_datasets["train"].classes)  # Should be 5 (Diabetic Retinopathy classes)

# ✅ Load Pretrained Swin Transformer Model
model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)
model = model.to(device)

# ✅ Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ✅ Learning Rate Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# ✅ Enable Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# ✅ Early Stopping Configuration
early_stopping_patience = 5
best_loss = float("inf")
epochs_no_improve = 0

# ✅ Training Function with Classification Report Logging
def train_model(model, dataloaders, criterion, optimizer, num_epochs=20):
    global best_loss, epochs_no_improve  

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        wandb.log({"epoch": epoch + 1})  

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

                # Store predictions & labels for classification report
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Compute epoch loss & accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = correct.double() / total

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # ✅ Compute & Log Classification Report (Only for Validation)
            if phase == "val":
                class_report = classification_report(
                    all_labels, all_preds, target_names=image_datasets["train"].classes, output_dict=True
                )

                # Log each class's precision, recall, f1-score
                for class_name, metrics in class_report.items():
                    if isinstance(metrics, dict):  # Skip overall averages
                        wandb.log({
                            f"{phase}_precision_{class_name}": metrics["precision"],
                            f"{phase}_recall_{class_name}": metrics["recall"],
                            f"{phase}_f1_{class_name}": metrics["f1-score"],
                        })

            # ✅ Log metrics to W&B
            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_accuracy": epoch_acc, "epoch": epoch + 1}, commit=True)

            # ✅ Update scheduler only after validation phase
            if phase == "val":
                scheduler.step(epoch_loss)

                # ✅ Early Stopping Logic
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), "best_swin_model.pth")  # Save best model
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered!")
                    return model  # Stop training early

        # Free GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    print("Training complete!")
    return model

# ✅ Train the Model
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=20)

# ✅ Save Final Model
torch.save(model.state_dict(), "swin_final_model.pth")
print("Model saved successfully!")

# ✅ Finish W&B Logging
wandb.finish()