import os
import shutil
import random
import cv2
import numpy as np

# ==========================================
# 1. PATH CONFIGURATIONS (Absolute)
# ==========================================
BASE_DIR = r"C:\Users\sadat\OneDrive\Desktop\Conference"
RAW_DATA_DIR = os.path.join(BASE_DIR, "dataset")
PREPARED_DIR = os.path.join(BASE_DIR, "prepared_dataset")
SHOWCASE_DIR = os.path.join(BASE_DIR, "showcase_train")
TRAIN_READY_DIR = os.path.join(BASE_DIR, "train_ready")

# Updated Split: 80% Train, 10% Val, 10% Test
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


# ==========================================
# 2. AUGMENTATION DEFINITIONS (10 Real-World)
# ==========================================
def augment_image(img):
    """Applies the 10 selected real-world augmentations to an image."""
    h, w = img.shape[:2]
    augs = {}

    # 1. Flipped Horizontal
    augs['flipped_horizontal'] = cv2.flip(img, 1)

    # 2. Flipped Vertical
    augs['flipped_vertical'] = cv2.flip(img, 0)

    # 3. Rotated (45 degrees)
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
    augs['rotated'] = cv2.warpAffine(img, M_rot, (w, h))

    # 4. Cropped (Center 80% resized back to original)
    crop_h, crop_w = int(h * 0.8), int(w * 0.8)
    start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    augs['cropped'] = cv2.resize(cropped, (w, h))

    # 5. Translated (Shifted x+50, y+50)
    M_trans = np.float32([[1, 0, 50], [0, 1, 50]])
    augs['translated'] = cv2.warpAffine(img, M_trans, (w, h))

    # 6. Brightness Adjusted
    augs['brightness_adjusted'] = cv2.convertScaleAbs(img, alpha=1, beta=30)

    # 7. Contrast Adjusted
    augs['contrast_adjusted'] = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    # 8. Color Jitter (HSV Shift)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    # Add flat random shifts to S and V channels (-30 to 30)
    hsv[:, :, 1] += np.random.randint(-30, 31)
    hsv[:, :, 2] += np.random.randint(-30, 31)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    augs['color_jitter'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 9. Gaussian Noise (Low light sensor grain)
    noise = np.random.normal(0, 25, img.shape).astype(np.float32)
    noisy_img = cv2.add(img.astype(np.float32), noise)
    augs['gaussian_noise'] = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # 10. Unsharp Mask
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    augs['unsharp_mask'] = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return augs


# ==========================================
# 3. PIPELINE EXECUTION
# ==========================================
def main():
    random.seed(42)  # For reproducible splits

    classes = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

    print("Step 1: Splitting dataset into 80/10/10 (Train/Val/Test)...")
    for cls in classes:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        train_idx = int(len(images) * SPLIT_RATIOS["train"])
        val_idx = train_idx + int(len(images) * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:train_idx],
            "val": images[train_idx:val_idx],
            "test": images[val_idx:]
        }

        # Copy to prepared_dataset
        for split_name, imgs in splits.items():
            split_dir = os.path.join(PREPARED_DIR, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in imgs:
                shutil.copy(os.path.join(cls_dir, img_name), os.path.join(split_dir, img_name))

    print("Step 2 & 3: Generating Showcase and PyTorch-Ready Datasets from Train Split...")
    train_base = os.path.join(PREPARED_DIR, "train")

    for cls in classes:
        cls_train_dir = os.path.join(train_base, cls)
        train_images = os.listdir(cls_train_dir)

        # Setup Track B (Flat PyTorch Directory)
        pytorch_cls_dir = os.path.join(TRAIN_READY_DIR, cls)
        os.makedirs(pytorch_cls_dir, exist_ok=True)

        for img_name in train_images:
            img_path = os.path.join(cls_train_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            base_name, ext = os.path.splitext(img_name)

            # --- Save Original to Track A (Showcase) ---
            showcase_orig_dir = os.path.join(SHOWCASE_DIR, cls, "original")
            os.makedirs(showcase_orig_dir, exist_ok=True)
            cv2.imwrite(os.path.join(showcase_orig_dir, img_name), img)

            # --- Save Original to Track B (PyTorch Flat) ---
            cv2.imwrite(os.path.join(pytorch_cls_dir, f"{base_name}_original{ext}"), img)

            # --- Generate & Save Augmentations ---
            augmented_images = augment_image(img)

            for aug_name, aug_img in augmented_images.items():
                # Track A (Showcase subdirectories)
                showcase_aug_dir = os.path.join(SHOWCASE_DIR, cls, aug_name)
                os.makedirs(showcase_aug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(showcase_aug_dir, img_name), aug_img)

                # Track B (PyTorch Flat)
                flat_aug_name = f"{base_name}_{aug_name}{ext}"
                cv2.imwrite(os.path.join(pytorch_cls_dir, flat_aug_name), aug_img)

        print(f"Processed {cls} - Expanded {len(train_images)} images into {len(train_images) * 11}.")

    print("\n✅ Dataset Pipeline Complete!")
    print(f"- Unaugmented Splits: {PREPARED_DIR}/")
    print(f"- Visual Showcase (For Paper): {SHOWCASE_DIR}/")
    print(f"- PyTorch Training Folder: {TRAIN_READY_DIR}/")


if __name__ == "__main__":
    main()