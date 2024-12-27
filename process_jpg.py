# This is for getting the results of baseline method by applying jpeg compression
import os
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import cv2
import shutil

data_dir = "./PlantVillage-Dataset/raw/color_copy"

categories = {
    "healthy": 0,  
    "disease": 1  
}

data = []
for category, label in categories.items():
    category_path = os.path.join(data_dir, category)
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        for file in os.listdir(subfolder_path):
            if file.endswith(".JPG") or file.endswith(".jpg"):
                file_path = os.path.join(subfolder_path, file)
                data.append({"file_path": file_path, "label": label})

# Pandas DataFrame
df = pd.DataFrame(data)
print(df.head())
print("Unique labels and their counts:")
print(df['label'].value_counts())

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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = LeafDataset(train_df, transform=transform)
val_dataset = LeafDataset(val_df, transform=transform)
test_dataset = LeafDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def get_binary_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
binary_model = get_binary_model()
binary_model = binary_model.to(device)

binary_criterion = nn.BCEWithLogitsLoss()
optimizer_binary = optim.Adam(binary_model.parameters(), lr=0.0001)

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10, patience=3):
    best_val_accuracy = 0.0
    best_model_wts = model.state_dict()
    epoch_no_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

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
    torch.save(model.state_dict(), "./weights/best_binary_model_mobile.pth")
    print("Best binary model saved as 'best_binary_model.pth'.")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer = dict([*self.model.named_modules()])[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, inputs):
        self.model.zero_grad()
        
        output = self.model(inputs).squeeze(1)
        
        output.backward()

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        cam = torch.mul(self.activations, pooled_gradients)
        cam = cam.sum(dim=1, keepdim=True)
        
        cam = torch.relu(cam)
        
        if cam.max() != 0:
            cam = cam / cam.max()
            
        return cam.squeeze()

# mask apply (No need)
def apply_mask(image, mask):
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = Image.fromarray((mask * 255).astype('uint8')).resize(image.size, Image.LANCZOS)
    mask = np.array(mask)
    
    image_array = np.array(image)
    threshold = 0.5
    binary_mask = (mask > threshold * 255).astype(np.float32)  
    binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
    
    binary_mask = np.stack([binary_mask] * 3, axis=-1)
    
    black_background = np.zeros_like(image_array)
    
    result = image_array * binary_mask + black_background * (1 - binary_mask)
    result = result.astype(np.uint8)
    
    return Image.fromarray(result)

def get_disease_name_from_path(file_path):
    parts = file_path.split(os.sep)
    if "disease" in parts:
        disease_idx = parts.index("disease")
        if len(parts) > disease_idx + 1:
            return parts[disease_idx + 1]
    return "unknown"

def evaluate_model(model, test_loader):
    all_labels = []
    all_preds = []
    
    grad_cam = GradCAM(model, target_layer="features.18")
    disease_dirs = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze(1)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            for idx, pred in enumerate(predicted):
                if pred.item() == 1:
                    global_idx = batch_idx * test_loader.batch_size + idx
                    if global_idx < len(test_loader.dataset):
                        original_image_path = test_loader.dataset.dataframe.iloc[global_idx]['file_path']
                        disease_name = get_disease_name_from_path(original_image_path)

                        with torch.enable_grad():
                            image = images[idx].unsqueeze(0).requires_grad_(True)
                            mask = grad_cam.generate_heatmap(image)

                        original_image = Image.open(original_image_path).convert('RGB')
                        # JPEG compression
                        compression_rates = [10, 30, 50, 70, 90]

                        # save images
                        for rate in compression_rates:
                            save_dir = os.path.join('compressed_images', f'quality_{rate}', disease_name)
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"image_{global_idx}_gradcam_quality_{rate}.jpg")
                            original_image.save(save_path, 'JPEG', quality=rate)

                        


    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    print("Starting training...")
    train_model(binary_model, binary_criterion, optimizer_binary, train_loader, val_loader, epochs=20, patience=3)
    
    print("\nLoading best model for evaluation...")
    binary_model.load_state_dict(torch.load("./weights/best_binary_model_mobile.pth"))
    binary_model.eval()
    
    print("\nStarting evaluation...")
    evaluate_model(binary_model, test_loader)