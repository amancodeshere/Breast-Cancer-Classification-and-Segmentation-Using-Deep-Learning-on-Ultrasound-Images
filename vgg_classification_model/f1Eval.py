from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from torchvision import models, transforms
import torch.nn as nn
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
model.classifier[6] = nn.Linear(4096, 3)
model.load_state_dict(torch.load('vgg19_breast_cancer_model.pth'))

# Set model to evaluation mode
model.eval()

data_dir = 'Dataset_BUSI_with_GT'

image_paths = []
labels = []

for label in os.listdir(data_dir) :
    class_folder = os.path.join(data_dir, label)
    if os.path.isdir(class_folder) :
        for image in os.listdir(class_folder) :
            if 'mask' not in image :
                image_paths.append(os.path.join(class_folder, image))
                labels.append(label)

# Create a DataFrame
data = pd.DataFrame({'filename': image_paths, 'label': labels})

label_mapping = {'benign': 0, 'malignant': 1, 'normal': 2}

x_train, x_test, y_train, y_test = train_test_split(
    data['filename'], data['label'], test_size=0.1, stratify=data['label'], random_state=42
)

y_train, y_test = y_train.map(label_mapping).astype(int), y_test.map(label_mapping).astype(int)

class ImageDataset(Dataset) :
    def __init__(self, image_path, labels, transform=None) :
        self.image_path = image_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx) :
        img_path = self.image_path[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
    
# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

# Load datasets
train_dataset = ImageDataset(image_path=x_train.tolist(), labels=y_train.tolist(), transform=transform)
test_dataset = ImageDataset(image_path=x_test.tolist(), labels=y_test.tolist(), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lists to hold all predictions and labels
all_preds = []
all_labels = []

# Disable gradient calculation for inference
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass to get predictions
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Collect predictions and labels for F1 calculation
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate F1 score
f1 = f1_score(all_labels, all_preds, average='weighted')  # use 'weighted' for multi-class
print(f'Weighted F1 Score: {f1:.4f}')
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()