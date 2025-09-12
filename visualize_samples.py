import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    labels_path = os.path.join(args.data_dir, "labels.csv")
    images_dir = os.path.join(args.data_dir, "images")
    df = pd.read_csv(labels_path).sample(n=min(args.k, len(pd.read_csv(labels_path))), random_state=0)

    cols = 4
    rows = (len(df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, df.iterrows()):
        img = imread(os.path.join(images_dir, row["filename"]), as_gray=True)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{row['filename']}\nlabel={row['label']}")
        ax.axis("off")

    # Hide any extra axes
    for j in range(len(df), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "sample_grid.png")
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved sample grid to {out_path}")

if __name__ == "__main__":
    main()
