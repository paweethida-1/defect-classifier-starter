import os
import csv
import argparse
import numpy as np
from skimage.draw import line, disk
from skimage.util import random_noise
from skimage.io import imsave

def make_blank(size):
    # Start from nearly uniform gray with mild illumination gradient
    base = np.full((size, size), 0.6, dtype=np.float32)
    gx = np.linspace(-0.05, 0.05, size, dtype=np.float32)
    gy = np.linspace(-0.05, 0.05, size, dtype=np.float32)
    base += gy[:, None] + gx[None, :]
    base = np.clip(base, 0.0, 1.0)
    return base

def add_scratch(img, thickness=2):
    h, w = img.shape
    x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
    x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
    rr, cc = line(y0, x0, y1, x1)
    for t in range(-thickness, thickness + 1):
        rr_t = np.clip(rr + t, 0, h - 1)
        cc_t = np.clip(cc + t, 0, w - 1)
        img[rr_t, cc_t] = np.clip(img[rr_t, cc_t] - np.random.uniform(0.3, 0.6), 0.0, 1.0)
    return img

def add_pit(img, radius=6):
    h, w = img.shape
    r, c = np.random.randint(radius, h - radius), np.random.randint(radius, w - radius)
    rr, cc = disk((r, c), radius)
    img[rr, cc] = np.clip(img[rr, cc] - np.random.uniform(0.2, 0.5), 0.0, 1.0)
    return img

def add_blob(img, radius=5):
    h, w = img.shape
    r, c = np.random.randint(radius, h - radius), np.random.randint(radius, w - radius)
    rr, cc = disk((r, c), radius)
    img[rr, cc] = np.clip(img[rr, cc] + np.random.uniform(0.2, 0.4), 0.0, 1.0)
    return img

def synth_image(size, defect=False):
    img = make_blank(size)
    # Add mild texture & noise
    img = random_noise(img, mode='gaussian', var=np.random.uniform(0.001, 0.005))
    img = np.clip(img, 0.0, 1.0)

    if defect:
        # Randomly choose one or two defect types
        kinds = ["scratch", "pit", "blob"]
        np.random.shuffle(kinds)
        k = np.random.choice([1, 2], p=[0.7, 0.3])
        for kind in kinds[:k]:
            if kind == "scratch":
                img = add_scratch(img, thickness=np.random.randint(1, 3))
            elif kind == "pit":
                img = add_pit(img, radius=np.random.randint(4, 8))
            elif kind == "blob":
                img = add_blob(img, radius=np.random.randint(4, 7))

    return np.clip(img, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic surface images + labels.csv")
    ap.add_argument("--n", type=int, default=400, help="total number of images")
    ap.add_argument("--size", type=int, default=100, help="image size (HxW)")
    ap.add_argument("--out", type=str, default="data", help="output dir (will create images/ and labels.csv)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    n_defect = args.n // 2
    n_clean = args.n - n_defect

    rows = []
    counter = 0

    for i in range(n_clean):
        img = synth_image(args.size, defect=False)
        fname = f"img_{counter:05d}.png"
        imsave(os.path.join(img_dir, fname), (img * 255).astype(np.uint8))
        rows.append((fname, 0))
        counter += 1

    for i in range(n_defect):
        img = synth_image(args.size, defect=True)
        fname = f"img_{counter:05d}.png"
        imsave(os.path.join(img_dir, fname), (img * 255).astype(np.uint8))
        rows.append((fname, 1))
        counter += 1

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for r in rows:
            w.writerow(r)

    print(f"Saved {len(rows)} images to {img_dir}")
    print(f"Saved labels to {labels_path}")

if __name__ == "__main__":
    main()
