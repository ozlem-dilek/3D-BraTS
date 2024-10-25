from google.colab import drive
drive.mount('/content/drive')

!pip install nibabel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import glob
import torch.nn.functional as F
from tqdm import tqdm

torch.cuda.empty_cache()


flair = glob.glob("/path/**/*flair.nii", recursive=True)

labels = glob.glob("/path/**/*seg.nii", recursive=True)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        # encoder kısmı
        self.encoder1 = double_conv(in_channels, 64)
        self.encoder2 = double_conv(64, 128)
        self.encoder3 = double_conv(128, 256)
        self.encoder4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # decoder kısmı
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = double_conv(128, 64)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        conv1 = self.encoder1(x)
        x = self.maxpool(conv1)

        conv2 = self.encoder2(x)
        x = self.maxpool(conv2)

        conv3 = self.encoder3(x)
        x = self.maxpool(conv3)

        x = self.encoder4(x)

        # decoder
        x = self.upconv3(x)
        diff3 = conv3.size()[2] - x.size()[2]
        diff3_h = conv3.size()[3] - x.size()[3]
        diff3_w = conv3.size()[4] - x.size()[4]
        x = F.pad(x, [diff3_w//2, diff3_w-diff3_w//2,
                     diff3_h//2, diff3_h-diff3_h//2,
                     diff3//2, diff3-diff3//2])
        x = torch.cat([x, conv3], dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        diff2 = conv2.size()[2] - x.size()[2]
        diff2_h = conv2.size()[3] - x.size()[3]
        diff2_w = conv2.size()[4] - x.size()[4]
        x = F.pad(x, [diff2_w//2, diff2_w-diff2_w//2,
                     diff2_h//2, diff2_h-diff2_h//2,
                     diff2//2, diff2-diff2//2])
        x = torch.cat([x, conv2], dim=1)
        x = self.decoder2(x)

        x = self.upconv1(x)
        diff1 = conv1.size()[2] - x.size()[2]
        diff1_h = conv1.size()[3] - x.size()[3]
        diff1_w = conv1.size()[4] - x.size()[4]
        x = F.pad(x, [diff1_w//2, diff1_w-diff1_w//2,
                     diff1_h//2, diff1_h-diff1_h//2,
                     diff1//2, diff1-diff1//2])
        x = torch.cat([x, conv1], dim=1)
        x = self.decoder1(x)

        x = self.final_conv(x)
        return x

class BRATSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_size=(155, 256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Görüntüleri yükle
        image = nib.load(self.image_paths[idx]).get_fdata()  # (240, 240, 155)
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())

        image = torch.from_numpy(image).float()  # (240, 240, 155)
        mask = torch.from_numpy(mask).long()

        # önce batch ve channel boyutlarını ekliyporum (N, C, D, H, W formatı için)
        image = image.unsqueeze(0).unsqueeze(0)  #(1, 1, 240, 240, 155)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # depth, height ve width boyutlarını yer değiştiriyorum
        image = image.permute(0, 1, 4, 2, 3)     # (1, 1, 155, 240, 240)
        mask = mask.permute(0, 1, 4, 2, 3)

        image = F.interpolate(image, size=self.target_size, mode='trilinear', align_corners=False)
        mask = F.interpolate(mask.float(), size=self.target_size, mode='nearest').long()

        return (
            image.squeeze(0),  # final shape'i böyle olması gerekiyo: (1, 155, 256, 256)
            mask.squeeze(0)
        )

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc='Training')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def main():
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 10

    image_paths = flair[:10]
    mask_paths = labels[:10]

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_dataset = BRATSDataset(
        train_images,
        train_masks,
        target_size=(64, 128, 128)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(in_channels=1, out_channels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    try:
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')

            # Train
            train_loss = train(model, train_loader, criterion, optimizer, device)
            print(f'Training Loss: {train_loss:.4f}')

            torch.cuda.empty_cache()

            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, f'model_epoch_{epoch+1}.pth')

    except Exception as e:
        print(f"Training error: {str(e)}")
        torch.cuda.empty_cache()

    print('Training completed!')

if __name__ == '__main__':
    main()

