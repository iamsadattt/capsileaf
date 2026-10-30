import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import multiprocessing

# ==========================================
# 1. CONFIGURATION & PATHS (Absolute Directories)
# ==========================================
BASE_DIR = r"C:\Users\sadat\OneDrive\Desktop\Conference"

TRAIN_DIR = os.path.join(BASE_DIR, "train_ready")
VAL_DIR = os.path.join(BASE_DIR, "prepared_dataset", "val")

# Renamed to protect your previous 74% baseline
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "conference_capsileaf_6_class_lr001.pth")
PLOT_SAVE_PATH = os.path.join(BASE_DIR, "conference_learning_curves_lr001.png")

BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_CLASSES = 6
LEARNING_RATE = 0.001  # Lowered for stable Adam fine-tuning


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # ==========================================
    # 2. DATA LOADERS (No inline augmentation)
    # ==========================================
    base_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=base_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=base_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # ==========================================
    # 3. MODEL ARCHITECTURE
    # ==========================================
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
                dataset_size = len(train_dataset)
            else:
                model.eval()
                dataloader = val_loader
                dataset_size = len(val_dataset)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, MODEL_SAVE_PATH)
                    print(f"*** New best model saved with Val Acc: {best_acc:.4f} ***")

    # FORCE SAVE THE FINAL WEIGHTS IF IT NEVER IMPROVED
    if best_acc == 0.0:
        print("*** Forcing save of final model weights (Validation accuracy did not improve) ***")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")

    # ==========================================
    # 5. GENERATE LEARNING CURVES
    # ==========================================
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"Learning curves saved to {PLOT_SAVE_PATH}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()