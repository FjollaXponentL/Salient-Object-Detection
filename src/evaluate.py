# src/evaluate.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from sod_model import SOD_CNN

from sklearn.metrics import precision_score, recall_score, f1_score

# -------- Settings --------
ROOT_PATH = "data/ECSSD"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint i ruajtur nga train.py
CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pth")


def compute_iou(y_true, y_pred):
    """
    Llogarit IoU (Intersection over Union) për vlera binare (0/1).
    """
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    intersection = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()

    if union == 0:
        return 0.0

    return intersection / (union + 1e-7)


def visualize_sample(image, mask, pred_mask, save_path=None):
    """
    Shfaq/shkruan një shembull: input image, ground truth mask, predicted mask, overlay.
    image: (C, H, W) tensor
    mask, pred_mask: (1, H, W) tensor
    """
    img_np = image.permute(1, 2, 0).cpu().numpy()
    gt_np = mask.squeeze().cpu().numpy()
    pred_np = pred_mask.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    # Overlay: image + predicted mask
    axes[3].imshow(img_np, alpha=0.7)
    axes[3].imshow(pred_np, cmap="jet", alpha=0.3)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    # dataLoader (test set
    _, _, test_loader = create_dataloaders(ROOT_PATH, batch_size=BATCH_SIZE)

    # model - checkpoint
    model = SOD_CNN().to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PATH}")

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # Nëse checkpoint-i ka key "model_state_dict"
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    all_preds = []
    all_targets = []

    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)       
            preds = (outputs > 0.5).float()  

            all_preds.append(preds.view(-1).cpu().numpy())
            all_targets.append(masks.view(-1).cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    iou = compute_iou(all_targets, all_preds)

    print("testingmetrics")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    
    os.makedirs("results", exist_ok=True)

    max_examples = 20   
    count = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            batch_size = images.size(0)

            for i in range(batch_size):
                save_path = os.path.join("results", f"example_{count + 1}.png")
                visualize_sample(
                    images[i].cpu(),
                    masks[i].cpu(),
                    preds[i].cpu(),
                    save_path=save_path,
                )
                print(f"Saved visualization: {save_path}")
                count += 1

                if count >= max_examples:
                    break

            if count >= max_examples:
                break


if __name__ == "__main__":
    main()
