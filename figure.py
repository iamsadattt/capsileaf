import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration: Base Directory ---
# Updated to your new Conference folder
base_dir = r'C:\Users\sadat\OneDrive\Desktop\Conference\prepared_dataset'

# Updated to perfectly match the 6 classes your model actually trained on
classes = [
    '12masi_morich(black)',
    'dhani_morich',
    'kajoli_morich',
    'naga_bombai_morich',
    'tal_morich',
    'thigh_chilli_pepper'
]

print("=" * 60)
print(" PART 1: DATASET COUNTING FOR TABLE I")
print("=" * 60)

table_data = []
total_raw = 0
total_aug = 0
total_val = 0
total_test = 0

for cls in classes:
    train_path = os.path.join(base_dir, 'train', cls)
    val_path = os.path.join(base_dir, 'val', cls)
    test_path = os.path.join(base_dir, 'test', cls)

    train_files = os.listdir(train_path) if os.path.exists(train_path) else []

    # Since OpenCV pools raw + augmented into train_ready, we look at the raw 'train'
    # folder to get the exact original count, and multiply by 10 for the augmented count.
    r_count = len(train_files)
    a_count = r_count * 10  # 10 augmentations per image

    val_files = os.listdir(val_path) if os.path.exists(val_path) else []
    test_files = os.listdir(test_path) if os.path.exists(test_path) else []

    v_count = len(val_files)
    t_count = len(test_files)

    total_raw += r_count
    total_aug += a_count
    total_val += v_count
    total_test += t_count

    table_data.append((cls, r_count, a_count, v_count, t_count))
    print(f"Class: {cls:<25} | Raw Train: {r_count:3d} | Aug Train: {a_count:4d} | Val: {v_count:2d} | Test: {t_count:2d}")

print("-" * 60)
print(f"{'TOTALS':<27} | Raw: {total_raw:3d} | Aug: {total_aug:4d} | Val: {total_val:2d} | Test: {total_test:2d}")
print("=" * 60)

# Print clean, ready-to-paste LaTeX rows for Table I
print("\n[READY-TO-PASTE LATEX ROWS FOR TABLE I]:")
for cls, r, a, v, t in table_data:
    # Clean formatting for LaTeX special characters
    latex_cls = cls.replace('_', r'\_').replace('(', r'\(').replace(')', r'\)')
    print(f"\\textit{{{latex_cls}}} & {r} & {a} & {v} & {t} \\\\")
print(f"\\textbf{{Total Images}} & \\textbf{{{total_raw}}} & \\textbf{{{total_aug}}} & \\textbf{{{total_val}}} & \\textbf{{{total_test}}} \\\\")

print("\n" + "=" * 60)
print(" PART 2: GENERATING FIGURE 2 (OPENCV AUGMENTATION GRID)")
print("=" * 60)

sample_img_path = None
for cls in classes:
    test_cls_dir = os.path.join(base_dir, 'test', cls)
    if os.path.exists(test_cls_dir):
        imgs = [f for f in os.listdir(test_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            sample_img_path = os.path.join(test_cls_dir, imgs[0])
            break

if sample_img_path and os.path.exists(sample_img_path):
    orig_bgr = cv2.imread(sample_img_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.resize(orig_rgb, (224, 224))

    # 1. Unsharp Mask
    gaussian = cv2.GaussianBlur(orig_bgr, (0, 0), 2.0)
    unsharp = cv2.addWeighted(orig_bgr, 1.5, gaussian, -0.5, 0)
    unsharp_rgb = cv2.cvtColor(unsharp, cv2.COLOR_BGR2RGB)
    unsharp_rgb = cv2.resize(unsharp_rgb, (224, 224))

    # 2. Gaussian Noise
    noise = np.random.normal(0, 25, orig_bgr.shape).astype(np.float32)
    noisy_img = cv2.add(orig_bgr.astype(np.float32), noise)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    noisy_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    noisy_rgb = cv2.resize(noisy_rgb, (224, 224))

    # 3. Brightness Adjusted
    bright = cv2.convertScaleAbs(orig_bgr, alpha=1, beta=30)
    bright_rgb = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
    bright_rgb = cv2.resize(bright_rgb, (224, 224))

    # 4. Flipped Horizontal
    flipped = cv2.flip(orig_rgb, 1)

    # Plotting the Grid using Matplotlib
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    titles = ['(a) Original', '(b) Unsharp Mask', '(c) Gaussian Noise', '(d) Brightness', '(e) Flipped']
    images = [orig_rgb, unsharp_rgb, noisy_rgb, bright_rgb, flipped]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    output_filename = 'conference_figure2_grid.jpg'
    plt.savefig(output_filename, dpi=300)
    print(f"Successfully generated and saved '{output_filename}' in project root!")
    plt.show()
else:
    print("Warning: Could not find a sample image to generate Figure 2 grid.")