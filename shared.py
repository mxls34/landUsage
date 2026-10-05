"""Things every page uses: the loaded models, test images, scores."""
import hashlib
import io
import os
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import config
import loader

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]


@st.cache_resource(show_spinner="Loading model…")
def _load(file_name: str, modified: float, settings: str) -> loader.LoadedModel:
    return loader.load(loader.BASE_DIR / config.MODELS_FOLDER / file_name)


def _settings() -> str:
    """Part of the cache key: editing config.py reloads the models."""
    return repr((config.CLASS_NAMES, config.IMG_SIZE, config.MEAN, config.STD, config.MODEL_SETTINGS))


def get_models() -> list:
    """Every model in models/ — each file is loaded once, then reused."""
    return [_load(p.name, p.stat().st_mtime, _settings()) for p in loader.model_files()]


def label(name: str) -> str:
    return f"{config.CLASS_ICONS.get(name, '📍')} {name}"


def _key(text) -> str:
    """'Annual Crop' / 'annual_crop' / 'AnnualCrop' -> 'annualcrop'."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


# ----------------------------------------------------------------- test images (dashboard)
@st.cache_data(show_spinner=False, ttl=10)
def scan_test_folder(folder: str, class_names: tuple, max_per_class):
    """Images in <folder>/<class name>/... -> (files, labels, skipped folder names)."""
    index = {_key(name): i for i, name in enumerate(class_names)}
    per_class = {i: [] for i in range(len(class_names))}
    skipped = set()
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        images = [f for f in filenames if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in IMAGE_TYPES]
        if not images:
            continue
        idx = index.get(_key(os.path.basename(dirpath)))
        if idx is None:
            skipped.add(os.path.basename(dirpath))
            continue
        per_class[idx] += [os.path.join(dirpath, f) for f in images]

    rng = random.Random(42)
    files, labels = [], []
    for idx, paths in per_class.items():
        paths.sort()
        if max_per_class and len(paths) > max_per_class:
            paths = sorted(rng.sample(paths, max_per_class))
        files += paths
        labels += [idx] * len(paths)
    return files, labels, sorted(skipped)


def extract_zip(data: bytes) -> str:
    """Unzip an uploaded test set once (same zip -> same folder)."""
    target = Path(tempfile.gettempdir()) / f"landusage_test_{hashlib.sha256(data).hexdigest()[:16]}"
    if not target.exists():
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(target)
    return str(target)


@st.cache_data(show_spinner=False, max_entries=50)
def evaluate(file_name: str, modified: float, settings: str, files: tuple) -> tuple:
    """Probabilities for every test image + ms per image (cached: each model is tested once)."""
    lm = _load(file_name, modified, settings)
    chunks, ms = [], 0.0
    for start in range(0, len(files), config.BATCH_SIZE):
        images = []
        for f in files[start:start + config.BATCH_SIZE]:
            with Image.open(f) as img:
                images.append(img.convert("RGB"))
        probs, t = loader.predict_batch(lm, images)
        chunks.append(probs)
        ms += t
    return np.concatenate(chunks), ms / max(len(files), 1)


def evaluate_model(lm: loader.LoadedModel, files: list) -> tuple:
    path = loader.BASE_DIR / config.MODELS_FOLDER / lm.file
    return evaluate(lm.file, path.stat().st_mtime, _settings(), tuple(files))


def score(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> dict:
    """Accuracy + macro precision / recall / F1 + confusion matrix."""
    cm = np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    present = support > 0
    return {"Accuracy": tp.sum() / max(cm.sum(), 1), "Precision": precision[present].mean(),
            "Recall": recall[present].mean(), "F1-score": f1[present].mean(),
            "cm": cm, "f1_per_class": f1, "support": support}
