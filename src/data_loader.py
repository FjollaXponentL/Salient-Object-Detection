# src/data_loader.py

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A  # për augmentations
from albumentations.pytorch import ToTensorV2

class ECSSD_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        # Sigurohu që imazhet dhe maskat përputhen
        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load image & mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # shape: (1, 128, 128)

        # Apply augmentations if defined
        # if self.transform:
        #     augmented = self.transform(image=image, mask=mask)
        #     image = augmented["image"]
        #     mask = augmented["mask"]

        # Convert image to CHW format for PyTorch
        image = np.transpose(image, (2, 0, 1))

        return image, mask


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])


def create_dataloaders(root_path, batch_size=4):
    # Folders for train, val, test
    train_img = os.path.join(root_path, "train/images")
    train_mask = os.path.join(root_path, "train/masks")

    val_img = os.path.join(root_path, "val/images")
    val_mask = os.path.join(root_path, "val/masks")

    test_img = os.path.join(root_path, "test/images")
    test_mask = os.path.join(root_path, "test/masks")

    # Create datasets
    train_dataset = ECSSD_Dataset(train_img, train_mask, transform=get_transforms(train=True))
    val_dataset = ECSSD_Dataset(val_img, val_mask, transform=get_transforms(train=False))
    test_dataset = ECSSD_Dataset(test_img, test_mask, transform=get_transforms(train=False))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


# Sanity check: vizualizimi i një batch imazhe + maska
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_loader, val_loader, test_loader = create_dataloaders("data/ECSSD", batch_size=4)

    print("Number of training batches:", len(train_loader))
    images, masks = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)

    # Visualize first image & mask
    img = images[0].numpy().transpose(1,2,0)
    mask = masks[0][0].numpy()

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Image")
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.show()
