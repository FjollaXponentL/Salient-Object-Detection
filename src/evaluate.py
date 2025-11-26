# src/evaluate.py
import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import create_dataloaders
from sod_model import SOD_CNN
from sklearn.metrics import precision_score, recall_score, f1_score

# -------- Settings --------
ROOT_PATH = "data/ECSSD"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.5  # threshold për binarizim
BATCH_SIZE = 1

# -------- Load Data --------
_, _, test_loader = create_dataloaders(ROOT_PATH, batch_size=BATCH_SIZE)

# -------- Load Model --------
model = SOD_CNN().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -------- Prepare folders & CSV --------
output_viz_dir = "outputs/visualizations"
os.makedirs(output_viz_dir, exist_ok=True)
output_plot_dir = "outputs/plots"
os.makedirs(output_plot_dir, exist_ok=True)
csv_path = os.path.join(output_plot_dir, "test_metrics_per_image.csv")

# Write CSV header
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "iou", "precision", "recall", "f1"])

# -------- Metrics accumulators --------
ious, precisions, recalls, f1s = [], [], [], []

# -------- Evaluation Loop (per-image reporting) --------
with torch.no_grad():
    for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = images.float().to(DEVICE)
        masks = masks.float().to(DEVICE)

        # Forward pass
        outputs = model(images)

        # Ensure outputs match mask spatial size
        if outputs.shape[2:] != masks.shape[2:]:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

        # Binarize predictions
        pred_mask = (outputs > THRESHOLD).float()

        # Convert to 1D integer arrays for sklearn
        pred_flat = (pred_mask.cpu().numpy().flatten() > 0).astype(int)
        true_flat = (masks.cpu().numpy().flatten() > 0.5).astype(int)

        # Compute IoU
        intersection = (pred_flat * true_flat).sum()
        union = pred_flat.sum() + true_flat.sum() - intersection + 1e-6
        iou = float(intersection / union) if union > 0 else 1.0
        ious.append(iou)

        # Compute precision, recall, f1 (zero_division=1 returns 1 when no positive predictions/labels)
        precision = precision_score(true_flat, pred_flat, zero_division=1)
        recall = recall_score(true_flat, pred_flat, zero_division=1)
        f1 = f1_score(true_flat, pred_flat, zero_division=1)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Append per-image metrics to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, iou, precision, recall, f1])

        # Print per-image metrics + running averages
        running_iou = float(np.mean(ious))
        running_f1 = float(np.mean(f1s))
        print(f"[{idx+1}/{len(test_loader)}] iou={iou:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f} | "
              f"running_mean_iou={running_iou:.4f} running_mean_f1={running_f1:.4f}")

        # -------- Visualization (save each) --------
        img_np = np.transpose(images[0].cpu().numpy(), (1, 2, 0))
        gt_np = masks[0].cpu().numpy()[0]
        pred_np = pred_mask[0].cpu().numpy()[0]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.title("Input")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Ground Truth")
        plt.imshow(gt_np, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Predicted")
        plt.imshow(pred_np, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Overlay")
        overlay = img_np * 0.6 + np.stack([pred_np]*3, axis=-1) * 0.4
        overlay = np.clip(overlay, 0, 1)
        plt.imshow(overlay)
        plt.axis("off")

        save_path = os.path.join(output_viz_dir, f"viz_{idx:04d}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Show only the first example in an interactive window for quick visual check
        if idx == 0:
            plt.figure(figsize=(12,4))
            plt.subplot(1,4,1); plt.title("Input"); plt.imshow(img_np); plt.axis("off")
            plt.subplot(1,4,2); plt.title("Ground Truth"); plt.imshow(gt_np, cmap="gray"); plt.axis("off")
            plt.subplot(1,4,3); plt.title("Predicted"); plt.imshow(pred_np, cmap="gray"); plt.axis("off")
            plt.subplot(1,4,4); plt.title("Overlay"); plt.imshow(overlay); plt.axis("off")
            plt.show()

# -------- Final summary & bar plot --------
mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
mean_prec = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
mean_rec = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
mean_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

print("\n--- Final Test Summary ---")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Precision: {mean_prec:.4f}")
print(f"Mean Recall: {mean_rec:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")

metrics = {"IoU": mean_iou, "Precision": mean_prec, "Recall": mean_rec, "F1-Score": mean_f1}
plt.figure(figsize=(8,5))
bars = plt.bar(metrics.keys(), metrics.values(), color=["skyblue", "lightgreen", "salmon", "orange"])
plt.ylim(0, 1)
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
plt.title("Test Set Metrics")
plt.savefig(os.path.join(output_plot_dir, "test_metrics_barplot.png"))
plt.close()

print(f"Saved per-image CSV at: {csv_path}")
print(f"Saved visualizations in: {output_viz_dir}")
print(f"Saved barplot at: {os.path.join(output_plot_dir, 'test_metrics_barplot.png')}")
