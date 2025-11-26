# src/sod_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SOD_CNN(nn.Module):
    def __init__(self):
        super(SOD_CNN, self).__init__()

        # ---- Encoder ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128 -> 64
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64 -> 32
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32 -> 16
        )

        # ---- Decoder ----
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 64 -> 128
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Decoder
        d3 = self.dec3(e3)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        out = self.out_conv(d1)
        out = self.sigmoid(out)

        return out


# ---- Loss function: BCE + 0.5*(1 - IoU) ----
def bce_iou_loss(pred, target):
    bce = nn.BCELoss()(pred, target)

    # Compute IoU
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    loss = bce + 0.5 * (1 - iou)
    return loss


# ---- Sanity check ----
if __name__ == "__main__":
    model = SOD_CNN()
    x = torch.randn(2, 3, 128, 128)  # batch_size=2
    y = model(x)
    print("Output shape:", y.shape)  # duhet të jetë (2,1,128,128)
