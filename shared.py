"""Things every page uses: the loaded models and the saved predictions."""
import hashlib
import io

import streamlit as st
from PIL import Image

import config
import loader

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]


@st.cache_resource(show_spinner="Loading model…")
def _load(file_name: str, modified: float, settings: str) -> loader.LoadedModel:
    return loader.load(loader.BASE_DIR / config.MODELS_FOLDER / file_name)


def _settings() -> str:
    """Part of the cache key: editing config.py, loader.py or custom_models.py reloads the models."""
    code = "".join((loader.BASE_DIR / f).read_text(encoding="utf-8") for f in ("loader.py", "custom_models.py"))
    return repr((config.CLASS_NAMES, config.IMG_SIZE, config.MEAN, config.STD, config.MODEL_SETTINGS,
                 hashlib.sha1(code.encode()).hexdigest()))


def get_models() -> list:
    """Every model in models/ — each file is loaded once, then reused."""
    return [_load(p.name, p.stat().st_mtime, _settings()) for p in loader.model_files()]


def save_prediction(upload_id: str, file_name: str, image: Image.Image, results: list, final: str) -> None:
    """Keep every predicted picture (this browser session) for the Dashboard page."""
    history = st.session_state.setdefault("history", [])
    if any(h["id"] == upload_id for h in history):
        return
    thumb = image.copy()
    thumb.thumbnail((400, 400))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    entry = {
        "id": upload_id, "file": file_name, "thumb": buf.getvalue(), "final": final,
        "rows": [{"Model": m.name, "Architecture": m.arch, "Answer": m.classes[int(p.argmax())],
                  "Confidence": float(p.max()) * 100, "ms": ms, "probs": p, "classes": m.classes}
                 for m, p, ms in results],
    }
    history.append(entry)


def label(name: str) -> str:
    return f"{config.CLASS_ICONS.get(name, '📍')} {name}"
