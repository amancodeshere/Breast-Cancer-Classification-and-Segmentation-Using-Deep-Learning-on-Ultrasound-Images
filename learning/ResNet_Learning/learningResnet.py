from datasets import load_dataset

# Load dataset
dataset = load_dataset('gymprathap/Breast-Cancer-Ultrasound-Images-Dataset')
train_data = dataset['train']

import torch
from torchvision import transforms
import torch.nn.functional as F

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (can adjust based on dataset size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
])

# %%
# Example of how you can preprocess an image from the dataset
def preprocess_sample(sample):
    image = preprocess(sample['image'])
    mask = torch.tensor(sample['segmentation_mask'])  # Assuming 'segmentation_mask' is the mask field
    return image, mask

# Preprocess entire dataset
train_data = [(preprocess_sample(sample)) for sample in train_data]
#%%


#%%
import torch
from torchvision import transforms

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (can adjust based on dataset size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
])

# Example of how you can preprocess an image from the dataset
def preprocess_sample(sample):
    image = preprocess(sample['image'])
    mask = torch.tensor(sample['segmentation_mask'])  # Assuming 'segmentation_mask' is the mask field
    return image, mask

# Preprocess entire dataset
train_data = [(preprocess_sample(sample)) for sample in train_data]
#%%


#%%

# Loss function
def dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)  # Apply sigmoid to get mask probabilities
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

# Alternatively, Binary Cross-Entropy with logits
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
#%%

#%%
# Training loop
def train(model, train_data, criterion, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in train_data:
            images = images.unsqueeze(0)  # Add batch dimension
            masks = masks.unsqueeze(0)    # Add batch dimension
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_data)}")

# Train the model
train(model, train_data, criterion, optimizer, epochs=10)
