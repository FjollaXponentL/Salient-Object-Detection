# src/sod_model.py

import torch
import torch.nn as nn


class SOD_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        # Decoder-skip connection
        self.dec1 = nn.ConvTranspose2d(256, 128, 2, stride=2)  
        self.conv_dec1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.dec2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  
        self.conv_dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.dec3 = nn.ConvTranspose2d(64, 32, 2, stride=2)   
        self.conv_dec3 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.dec4 = nn.ConvTranspose2d(32, 16, 2, stride=2)    
        self.out = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)

        # Decoder + skip connections
        x = self.dec1(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.conv_dec1(x)

        x = self.dec2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv_dec2(x)

        x = self.dec3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv_dec3(x)

        x = self.dec4(x)
        x = self.out(x)

        return self.sigmoid(x)


def bce_iou_loss(pred, target):
    """Binary Cross Entropy + 0.5 * (1 - IoU)"""
    bce = nn.BCELoss()(pred, target)

    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    loss = bce + 0.5 * (1.0 - iou)
    return loss
