import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset_dir = "Dataset_BUSI_with_GT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epoch = 0
train_loss = []

class unetwork(nn.Module) :
    # Use threshold = 0.5 for binary mask
    def __init__(self):
        super(unetwork, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 256, 3, 1)
        self.convTrans1 = nn.ConvTranspose2d(256, 256, 3, 1)
        self.upsam = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(256, 64, 3, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1)
        self.convTrans2 = nn.ConvTranspose2d(32, 16, 3, 1)
        self.conv7 = nn.Conv2d(16, 4, 3 ,1)
        self.conv8 = nn.Conv2d(4, 2, 3, 1)
        self.convFin = nn.Conv2d(2, 1, 1, 1)

    def forward(self, x) :
        con = []
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        con.append(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        con.append(x)
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        con.append(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        con.append(x)
        x = self.pool(x)
        x = F.leaky_relu(self.convTrans1(x), negative_slope=0.01)
        con[3] = F.interpolate(con[3], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = x + con[3]
        x = self.upsam(x)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.01)
        con[2] = F.interpolate(con[2], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = x + con[2]
        x = self.upsam(x)
        x = F.leaky_relu(self.conv6(x), negative_slope=0.01)
        x = F.leaky_relu(self.convTrans2(x), negative_slope=0.01)
        con[1] = F.interpolate(con[1], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = x + con[1]
        x = self.upsam(x)
        x = F.leaky_relu(self.conv7(x), negative_slope=0.01)
        con[0] = F.interpolate(con[0], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = x + con[0]
        x = self.upsam(x)
        x = F.leaky_relu(self.conv8(x), negative_slope=0.01)
        x = F.leaky_relu(self.convFin(x), negative_slope=0.01)

        return x

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        # Traverse the dataset structure
        for class_folder in os.listdir(root_dir):
            if class_folder == "normal":
                continue
            class_folder_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_folder_path):
                for file in os.listdir(class_folder_path):
                    if (file.endswith(".png")) & ('mask' not in file):

                        image_path = os.path.join(class_folder_path, file)
                        mask_path = image_path.replace(".png", "_mask") + ".png"
                        mask_path2 = image_path.replace(".png", "_mask_1") + ".png"
                        mask_path3 = image_path.replace(".png", "_mask_2") + ".png"
                        if os.path.exists(mask_path):
                            self.image_paths.append(image_path)
                            if not os.path.exists(mask_path2):
                                mask_path2 = mask_path
                            if not os.path.exists(mask_path3):
                                mask_path3 = mask_path
                            self.mask_paths.append((mask_path, mask_path2, mask_path3))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i]).convert('L')
        mask1 = Image.open(self.mask_paths[i][0]).convert('L')
        mask2 = Image.open(self.mask_paths[i][1]).convert('L')
        mask3 = Image.open(self.mask_paths[i][2]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask1 = self.transform(mask1)
            mask2 = self.transform(mask2)
            mask3 = self.transform(mask3)
        
        mask1 = (mask1 > 0.5).float()
        mask2 = (mask2 > 0.5).float()
        mask3 = (mask3 > 0.5).float()
        mask = torch.max(mask1, torch.max(mask2, mask3))
        mask = (mask > 0.5).float()
        return image, mask

def iou(model_output, mask) :
    model_output = F.interpolate(model_output, size=(mask.size(1), mask.size(2)), mode = 'nearest')
    image = (model_output.squeeze(1) > 0.5).bool()
    mask = (mask.squeeze(1) > 0.5).bool()
    #print(f"input: {image}")
    #print(f"target: {mask}")
    #print(f"Mask shape: {mask.shape}, Output shape: {image.shape}")
    intersect = (image & mask).float().sum()
    union = (image | mask).float().sum()
    # Avoid divide by 0
    if union == 0:
        return torch.tensor(0.0)

    return intersect / union

def f1(model_output, mask) :
    model_output = F.interpolate(model_output, size=(mask.size(1), mask.size(2)), mode = 'nearest')
    model_output = (model_output.squeeze(1) > 0.5).bool()
    mask = (mask.squeeze(1) > 0.5).bool()
    TP = torch.sum(model_output & mask).float()
    FP = torch.sum(model_output & ~mask).float()
    FN = torch.sum(~model_output & mask).float()
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def test(model, device, test_loader) :
    model.eval()
    total_iou = 0.0
    total_f1 = 0.0
    num_test = 0

    with torch.no_grad() :
        for image, mask in test_loader :
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            threshold = torch.sigmoid(output)
            threshold = (threshold > 0.5).float()
            cur_iou = iou(image, mask)
            cur_f1 = f1(image, mask)
            print(f"IOU: {cur_iou}\tF1: {cur_f1}")
            total_iou += cur_iou
            total_f1 += cur_f1
            num_test += 1
            
    print('\nTest set: Average iou: {:.4f} Average f1: {:.4f} Within {} tests\n'.format(
        total_iou/len(test_loader.dataset), total_f1/len(test_loader.dataset), num_test))

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

dataset = ImageDataset(dataset_dir, transform)

train_sub, test_sub = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=36)

train_data = torch.utils.data.Subset(dataset, train_sub)
test_data = torch.utils.data.Subset(dataset, test_sub)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

model = unetwork().to(device)
model.load_state_dict(torch.load("modded_unet_Kong.pth", weights_only=True))
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

model.eval()

print("\nTesting Start -----------------------\n")
test(model, device, test_loader)

with torch.no_grad():
    for image, mask in test_loader:
        image, mask = image.to(device), mask.to(device)
        print(f"IOU: {iou(mask, mask)}")

        output = model(image)

        image = image[0].cpu().numpy().squeeze()
        mask = mask[0].cpu().numpy().squeeze()
        output = torch.sigmoid(output[0])
        output = (output > 0.5).float().cpu().numpy().squeeze()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Input Image')
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[2].imshow(output, cmap='gray')
        ax[2].set_title('Predicted Mask')

        plt.show()

        break