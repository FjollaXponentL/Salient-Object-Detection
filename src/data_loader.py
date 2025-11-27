# src/data_loader.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECSSD_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, train=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train = train

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))


        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

    
        if self.train:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

    
        image = np.transpose(image, (2, 0, 1))     
        mask = np.expand_dims(mask, axis=0)        

        
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask


def create_dataloaders(root_path, batch_size=8):
    train_img = os.path.join(root_path, "train/images")
    train_mask = os.path.join(root_path, "train/masks")

    val_img = os.path.join(root_path, "val/images")
    val_mask = os.path.join(root_path, "val/masks")

    test_img = os.path.join(root_path, "test/images")
    test_mask = os.path.join(root_path, "test/masks")

    train_dataset = ECSSD_Dataset(train_img, train_mask, train=True)
    val_dataset = ECSSD_Dataset(val_img, val_mask, train=False)
    test_dataset = ECSSD_Dataset(test_img, test_mask, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=1,          shuffle=False)

    return train_loader, val_loader, test_loader
