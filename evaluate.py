import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & PATHS (Absolute)
# ==========================================
BASE_DIR = r"C:\Users\sadat\OneDrive\Desktop\Conference"

TEST_DIR = os.path.join(BASE_DIR, "prepared_dataset", "test")

# Pointing to the NEW weights and a NEW confusion matrix name
MODEL_WEIGHTS = os.path.join(BASE_DIR, "conference_capsileaf_6_class_lr001.pth")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "conference_confusion_matrix_lr001.png")

NUM_CLASSES = 6
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATA LOADING & MODEL SETUP
# ==========================================
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = test_dataset.classes

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ==========================================
# 3. INFERENCE LOOP
# ==========================================
all_preds = []
all_labels = []

print(f"Evaluating model on {len(test_dataset)} unseen test images...")

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================================
# 4. METRICS & VISUALIZATION
# ==========================================
print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Test Set Confusion Matrix (LR=0.001)')
plt.ylabel('True Cultivar')
plt.xlabel('Predicted Cultivar')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")