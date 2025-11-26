# src/train.py

import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from data_loader import create_dataloaders
from sod_model import SOD_CNN, bce_iou_loss
from tqdm import tqdm

# -------- Settings --------
ROOT_PATH = "data/ECSSD"  # root path për dataset
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/sod_experiment")

# -------- DataLoader --------
train_loader, val_loader, test_loader = create_dataloaders(ROOT_PATH, batch_size=BATCH_SIZE)

# -------- Model --------
model = SOD_CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")

# -------- Training Loop --------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training"):
        images = images.float().to(DEVICE)
        masks = masks.float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = bce_iou_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().to(DEVICE)
            masks = masks.float().to(DEVICE)
            outputs = model(images)
            loss = bce_iou_loss(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # -------- Logging --------
    print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Val", avg_val_loss, epoch)

    # -------- Save checkpoint --------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss
        }, checkpoint_path)
        print(f"Saved best model checkpoint at epoch {epoch}")
