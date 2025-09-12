from __future__ import annotations
import numpy as np
from skimage.feature import hog
from skimage.filters import sobel, threshold_otsu
from skimage import img_as_float

def extract_features(image: np.ndarray, kind: str = "hog+stats") -> np.ndarray:
    """Return a 1D feature vector from a grayscale image in [0,1].
    kind: "hog", "stats", or "hog+stats"
    """
    if image.ndim == 3:
        # Convert to grayscale if needed (simple average)
        image = image.mean(axis=2)
    img = img_as_float(image)

    feats = []

    if "hog" in kind:
        hog_vec = hog(
            img,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(hog_vec)

    if "stats" in kind:
        m = float(np.mean(img))
        s = float(np.std(img))
        # Edge magnitude via Sobel
        sob = sobel(img)
        edge_density = float((sob > 0.05).mean())
        # High-gradient ratio using Otsu threshold on sobel
        try:
            thr = threshold_otsu(sob)
        except ValueError:
            thr = 0.0
        high_grad_ratio = float((sob > thr).mean())
        stats = np.array([m, s, edge_density, high_grad_ratio], dtype=np.float32)
        feats.append(stats)

    if not feats:
        raise ValueError("Unknown feature kind. Use 'hog', 'stats', or 'hog+stats'.")

    return np.concatenate(feats).astype(np.float32)
