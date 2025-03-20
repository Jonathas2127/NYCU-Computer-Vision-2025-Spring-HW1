import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import numpy as np
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.0001
NUM_CLASSES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation (Stronger to Reduce Overfitting)
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Dataset (Modify with actual paths)
train_dataset = datasets.ImageFolder(root='/kaggle/input/hw1-data/data/train', transform=transform_train)
val_dataset = datasets.ImageFolder(root='/kaggle/input/hw1-data/data/val', transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Save the class mapping
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}  # Reverse mapping

# Custom dataset for test images (no class subdirectories)
class TestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith(('jpg', 'jpeg', 'png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Return image and filename


test_dataset = TestImageDataset(root_dir='/kaggle/input/hw1-data/data/test', transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


# Modified ResNeXt with Dropout to Reduce Overfitting
class ResNeXtClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNeXtClassifier, self).__init__()
        self.resnext = timm.create_model('resnext50_32x4d', pretrained=True)
        self.resnext.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnext.num_features, num_classes)
        )

        for param in self.resnext.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnext(x)


model = ResNeXtClassifier().to(device)

# Loss & Optimizer with Stronger Regularization
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)


# Training Function with Accuracy Calculation
def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Apply Gradient Clipping (Prevents instability)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

        evaluate(epoch)


# Evaluation Function with Accuracy Calculation
def evaluate(epoch):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")


# Generate Predictions with Correct Label Mapping
def generate_predictions():
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, img_paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            class_name = idx_to_class[predicted.item()]  # Get actual class label
            image_name = os.path.splitext(os.path.basename(img_paths[0]))[0]
            predictions.append((image_name, class_name))

    df = pd.DataFrame(predictions, columns=['image_name', 'pred_label'])
    df.to_csv('prediction.csv', index=False)
    print("Predictions saved to prediction.csv")


# Running the pipeline
if __name__ == "__main__":
    train()
    generate_predictions()
