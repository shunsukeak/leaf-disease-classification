# This is for server-side process
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from torchvision import models

data_dir = "./processed_test_0.5"


# attach labels
subcategories = {folder: idx for idx, folder in enumerate(os.listdir(data_dir))}
print("Subcategory to Label Mapping:")
for subcategory, label in subcategories.items():
    print(f"{subcategory}: {label}")

# collect image data
data = []
for subfolder, label in subcategories.items():
    subfolder_path = os.path.join(data_dir, subfolder)
    print(f"Checking subfolder: {subfolder_path}")
    if not os.path.isdir(subfolder_path):
        print(f"Not a directory: {subfolder_path}")
        continue
    for file in os.listdir(subfolder_path):
        if file.endswith(".JPG") or file.endswith(".jpg"):
            file_path = os.path.join(subfolder_path, file)
            data.append({"file_path": file_path, "label": label})
            # print(f"Added file: {file_path}")

if not data:
    print("No files found matching the criteria.")

df = pd.DataFrame(data)

if df.empty:
    print("Error: DataFrame is empty after data collection.")
else:
    print("DataFrame head:")
    print(df.head())


# Split data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df["label"], random_state=42)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

class LeafDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

# data transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# create dataloader
train_dataset = LeafDataset(train_df, transform=transform)
val_dataset = LeafDataset(val_df, transform=transform)
test_dataset = LeafDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def get_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_model(num_classes=len(subcategories))

# optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10, patience=3):
    print("train start")
    best_val_accuracy = 0.0
    best_model_wts = model.state_dict()
    epoch_no_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict()
            epoch_no_improvement = 0
        else:
            epoch_no_improvement += 1

        if epoch_no_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "new_best_model_resnet50_multiclass.pth")

    print("Best model saved.")

# start training
# train_model(model, criterion, optimizer, train_loader, val_loader, epochs=50, patience=5)

# model evaluation
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, test_loader):
    print("Eval start")
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))


model.load_state_dict(torch.load("new_best_model_resnet50_multiclass.pth"))
model.eval()
evaluate_model(model, test_loader)
