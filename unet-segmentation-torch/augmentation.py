import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import numpy as np

# Custom dataset class
class BreastCancerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = [name for name in os.listdir(image_dir) if name.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)  # Assumes masks have the same name as images

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are single-channel grayscale

        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert mask to tensor with appropriate dtype
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# Transformation class to apply augmentations to both image and mask
class Transform:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, mask):
        augmented = self.augmentations(image=np.array(image), mask=np.array(mask))
        return Image.fromarray(augmented['image']), Image.fromarray(augmented['mask'])

# Directory paths
image_dir = "/path/to/images"
mask_dir = "/path/to/masks"

# Augmentations for the training set
train_transforms = Transform(
    augmentations=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])
)

# Transform for validation and test sets
val_test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load dataset
dataset = BreastCancerDataset(image_dir, mask_dir, transform=None)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Apply transformations
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")