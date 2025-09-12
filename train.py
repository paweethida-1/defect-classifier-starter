import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib

from src.features import extract_features

def load_dataset(data_dir: str):
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")
    df = pd.read_csv(labels_path)
    filepaths = [os.path.join(images_dir, fn) for fn in df["filename"].tolist()]
    labels = df["label"].astype(int).to_numpy()
    return filepaths, labels

def build_features(filepaths, feature_kind: str):
    X = []
    for fp in tqdm(filepaths, desc="Extracting features"):
        img = imread(fp, as_gray=True)
        feats = extract_features(img, kind=feature_kind)
        X.append(feats)
    X = np.vstack(X)
    return X

def plot_confusion_matrix(cm, classes, out_path: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--features", type=str, default="hog+stats", choices=["hog","stats","hog+stats"])
    ap.add_argument("--model", type=str, default="svm", choices=["svm","rf"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--outputs", type=str, default="outputs")
    ap.add_argument("--models-dir", type=str, default="models")
    args = ap.parse_args()

    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    filepaths, labels = load_dataset(args.data_dir)
    X = build_features(filepaths, args.features)
    y = labels

    X_train, X_test, y_train, y_test, fp_train, fp_test = train_test_split(
        X, y, filepaths, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if args.model == "svm":
        clf = SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=args.random_state)
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=args.random_state, n_jobs=-1)

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    # Metrics & plots
    cr = classification_report(y_test, y_pred, target_names=["normal (0)","defect (1)"])
    with open(os.path.join(args.outputs, "classification_report.txt"), "w") as f:
        f.write(cr)
    print(cr)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=["0","1"], out_path=os.path.join(args.outputs, "confusion_matrix.png"))

    # Save test split (for later evaluation) and model artifact
    pd.DataFrame({"filename": [os.path.basename(p) for p in fp_test], "label": y_test}).to_csv(
        os.path.join(args.outputs, "test_split.csv"), index=False
    )

    artifact = {
        "model": clf,
        "scaler": scaler,
        "feature_kind": args.features,
        "image_size": None,  # not strictly required since HOG is size-agnostic with our settings
    }
    model_name = f"model_{args.model}_{args.features}.pkl"
    joblib.dump(artifact, os.path.join(args.models_dir, model_name))
    print(f"Saved model to {os.path.join(args.models_dir, model_name)}")
    print(f"Saved confusion matrix to {os.path.join(args.outputs, 'confusion_matrix.png')}")
    print(f"Saved test split to {os.path.join(args.outputs, 'test_split.csv')}")

if __name__ == "__main__":
    main()
