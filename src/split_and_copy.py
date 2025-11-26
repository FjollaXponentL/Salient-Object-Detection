# src/split_and_copy.py
import os
import shutil

ROOT = "data/ECSSD"
IMG_DIR = os.path.join(ROOT, "images")
MASK_DIR = os.path.join(ROOT, "masks")

SPLITS = ["train", "val", "test"]

for split in SPLITS:
    txt_file = os.path.join(ROOT, f"{split}.txt")
    target_img_dir = os.path.join(ROOT, split, "images")
    target_mask_dir = os.path.join(ROOT, split, "masks")

    with open(txt_file, "r") as f:
        files = f.read().splitlines()

    for file_name in files:
        src_img = os.path.join(IMG_DIR, file_name)
        
        # ndrysho extension nga .jpg në .png për masks
        mask_name = os.path.splitext(file_name)[0] + ".png"
        src_mask = os.path.join(MASK_DIR, mask_name)

        dst_img = os.path.join(target_img_dir, file_name)
        dst_mask = os.path.join(target_mask_dir, mask_name)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_mask, dst_mask)

print("All images and masks copied to train/val/test folders successfully!")
