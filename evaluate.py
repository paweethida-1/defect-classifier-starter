import os
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

from src.features import extract_features

def predict_folder(model_artifact, folder: str, feature_kind: str):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    preds = []
    for fn in files:
        img = imread(os.path.join(folder, fn), as_gray=True)
        x = extract_features(img, kind=feature_kind).reshape(1, -1)
        x = model_artifact["scaler"].transform(x)
        y = model_artifact["model"].predict(x)[0]
        preds.append((fn, int(y)))
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="models/model_svm_hog+stats.pkl")
    ap.add_argument("--use-saved-test", action="store_true", help="Evaluate on outputs/test_split.csv")
    ap.add_argument("--predict-dir", type=str, help="Predict all images in a folder")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--outputs", type=str, default="outputs")
    args = ap.parse_args()

    artifact = joblib.load(args.model_path)
    feature_kind = artifact.get("feature_kind", "hog+stats")

    if args.use_saved_test:
        test_csv = os.path.join(args.outputs, "test_split.csv")
        if not os.path.exists(test_csv):
            raise FileNotFoundError("outputs/test_split.csv not found. Train first to create test split.")
        df = pd.read_csv(test_csv)
        imgs_dir = os.path.join(args.data_dir, "images")

        y_true, y_pred = [], []
        for _, row in df.iterrows():
            fn = row["filename"]
            y_true.append(int(row["label"]))
            img = imread(os.path.join(imgs_dir, fn), as_gray=True)
            x = extract_features(img, kind=feature_kind).reshape(1, -1)
            x = artifact["scaler"].transform(x)
            y = artifact["model"].predict(x)[0]
            y_pred.append(int(y))

        print(classification_report(y_true, y_pred, target_names=["normal (0)","defect (1)"]))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0","1"])
        fig, ax = plt.subplots(figsize=(4,4))
        disp.plot(ax=ax, colorbar=False)
        plt.tight_layout()
        os.makedirs(args.outputs, exist_ok=True)
        out_path = os.path.join(args.outputs, "confusion_matrix_eval.png")
        plt.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved evaluation confusion matrix to {out_path}")

    if args.predict_dir:
        preds = predict_folder(artifact, args.predict_dir, feature_kind)
        print("Predictions (filename, label):")
        for fn, y in preds:
            print(f"{fn}, {y}")

if __name__ == "__main__":
    main()
