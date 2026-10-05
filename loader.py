"""
Loads any PyTorch classification model in models/ and predicts one image.
You don't need to edit this file — change config.py instead.

A .pt file can be:
  * a whole model            torch.save(model, "best.pt")
  * only the weights         torch.save(model.state_dict(), "best.pt")
  * a checkpoint dictionary  torch.save({"model_state_dict": ..., "class_names": [...]}, "best.pt")
  * a TorchScript model      torch.jit.script(model).save("best.pt")
For "only the weights", the architecture (ResNet, MobileNetV3, ViT, EfficientNet, ...)
and the classifier head are worked out from the weights themselves.
"""
from __future__ import annotations

import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image

import config

BASE_DIR = Path(__file__).resolve().parent
MODEL_EXTS = {".pt", ".pth", ".pkl", ".bin"}

# torchvision architectures we can rebuild from weights only: name -> classifier attribute
ARCHITECTURES = {
    "resnet18": "fc", "resnet34": "fc", "resnet50": "fc", "resnet101": "fc", "resnet152": "fc",
    "mobilenet_v3_large": "classifier", "mobilenet_v3_small": "classifier", "mobilenet_v2": "classifier",
    "efficientnet_b0": "classifier", "efficientnet_b1": "classifier", "efficientnet_b2": "classifier",
    "efficientnet_b3": "classifier", "efficientnet_v2_s": "classifier",
    "convnext_tiny": "classifier", "convnext_small": "classifier",
    "densenet121": "classifier",
    "vit_b_16": "heads", "vit_b_32": "heads", "vit_l_16": "heads",
}
PRETTY = {
    "mobilenet_v3_large": "MobileNetV3-Large", "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v2": "MobileNetV2", "vit_b_16": "ViT-B/16", "vit_b_32": "ViT-B/32", "vit_l_16": "ViT-L/16",
}


@dataclass
class LoadedModel:
    name: str                      # file name without .pt
    file: str
    model: nn.Module | None = None
    arch: str = "?"
    classes: list = field(default_factory=list)
    img_size: int = config.IMG_SIZE
    error: str | None = None
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.model is not None


# =============================================================================
#  LOADING
# =============================================================================
def model_files() -> list[Path]:
    folder = BASE_DIR / config.MODELS_FOLDER
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in MODEL_EXTS)


def load(path: Path) -> LoadedModel:
    """Never raises: a broken file comes back with .error set."""
    settings = config.MODEL_SETTINGS.get(path.name, {})
    lm = LoadedModel(name=path.stem, file=path.name, img_size=settings.get("img_size", config.IMG_SIZE))
    try:
        obj = _read(path)
        model, state, info = _unwrap(obj)
        if model is None:
            arch = settings.get("arch") or info.get("arch") or _try_guess_state(state)
            if arch:
                model = _build(arch, state)
                lm.arch = _pretty(arch)
            else:
                model = _build_custom(state)
                lm.arch = getattr(model, "arch_name", type(model).__name__)
        else:
            lm.arch = _pretty(_try_guess(model) or type(model).__name__)
        model.eval()
        n = _count_outputs(model, lm.img_size)
        lm.classes = _class_names(settings.get("classes") or info.get("classes"), n, lm)
        lm.model = model
    except Exception as exc:  # one bad file must not break the website
        lm.error = _explain(exc)
    return lm


def _read(path: Path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.jit.load(str(path), map_location="cpu")       # TorchScript
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=True)   # weights / checkpoint
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)  # whole pickled model


def _unwrap(obj):
    """-> (model or None, state_dict or None, info about classes / architecture)."""
    info = {}
    if isinstance(obj, nn.Module):
        return obj, None, info
    if not isinstance(obj, dict):
        raise ValueError(f"Don't know how to read a {type(obj).__name__} saved in this file")

    for key in ("class_names", "classes", "labels"):
        if isinstance(obj.get(key), (list, tuple)):
            info["classes"] = [str(c) for c in obj[key]]
    if isinstance(obj.get("idx_to_class"), dict):
        info["classes"] = [str(obj["idx_to_class"][k]) for k in sorted(obj["idx_to_class"], key=int)]
    if isinstance(obj.get("class_to_idx"), dict):
        info["classes"] = [c for c, _ in sorted(obj["class_to_idx"].items(), key=lambda kv: kv[1])]
    for key in ("arch", "architecture", "model_name"):
        if isinstance(obj.get(key), str) and obj[key].lower() in ARCHITECTURES:
            info["arch"] = obj[key].lower()

    for key in ("model", "net"):
        if isinstance(obj.get(key), nn.Module):
            return obj[key], None, info
    state = obj
    for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
        if isinstance(obj.get(key), dict):
            state = obj[key]
            break
    state = OrderedDict((k, v) for k, v in state.items() if torch.is_tensor(v))
    if not state:
        raise ValueError("No model weights found in this file")
    return None, _strip_prefix(state), info


def _strip_prefix(state):
    """Remove 'module.' (DataParallel), '_orig_mod.' (torch.compile) or 'model.' from every key."""
    for prefix in ("module.", "_orig_mod.", "model."):
        if all(k.startswith(prefix) for k in state):
            state = OrderedDict((k[len(prefix):], v) for k, v in state.items())
    return state


@lru_cache(maxsize=None)
def _backbone_signature(arch: str) -> dict:
    """Shapes of every weight except the classifier (built on the 'meta' device: no memory used)."""
    head = ARCHITECTURES[arch]
    with torch.device("meta"):
        model = getattr(tvm, arch)()
    return {k: tuple(v.shape) for k, v in model.state_dict().items()
            if not k.startswith(head + ".") and k != head}


def _guess_architecture(state) -> str:
    for arch, head in ARCHITECTURES.items():
        body = {k: tuple(v.shape) for k, v in state.items() if not k.startswith(head + ".") and k != head}
        if body == _backbone_signature(arch):
            return arch
    raise ValueError("Could not recognise the architecture from the weights")


def _try_guess_state(state) -> str | None:
    try:
        return _guess_architecture(state)
    except ValueError:
        return None


def _build_custom(state) -> nn.Module:
    """Try the classes in custom_models.py (your own architectures)."""
    import custom_models
    matrices = [v for v in state.values() if v.dim() == 2]
    num_classes = matrices[-1].shape[0] if matrices else 1
    for cls in custom_models.CUSTOM_MODELS:
        try:
            model = cls(num_classes=num_classes)
            model.load_state_dict(state, strict=True)
            return model
        except Exception:
            continue
    raise ValueError("Could not recognise the architecture from the weights")


def _try_guess(model: nn.Module) -> str | None:
    try:
        return _guess_architecture(model.state_dict())
    except Exception:
        return None


def _build(arch: str, state) -> nn.Module:
    head = ARCHITECTURES[arch]
    model = getattr(tvm, arch)()
    head_weights = OrderedDict((k[len(head) + 1:], v) for k, v in state.items() if k.startswith(head + "."))
    setattr(model, head, _rebuild(getattr(model, head), head_weights))
    model.load_state_dict(state, strict=True)
    return model


def _rebuild(default: nn.Module | None, params: OrderedDict) -> nn.Module:
    """A module whose weights fit `params` (keys relative to the module).
    Keeps torchvision's own layers where they fit, so a usual fine-tuned head loads as-is."""
    if default is not None and _shapes(default.state_dict()) == _shapes(params):
        return default
    if set(params) <= {"weight", "bias", "running_mean", "running_var", "num_batches_tracked"}:
        return _leaf(params)

    groups = OrderedDict()
    for key, value in params.items():
        child, _, rest = key.partition(".")
        groups.setdefault(child, OrderedDict())[rest] = value
    defaults = dict(default.named_children()) if default is not None else {}
    if not all(name.isdigit() for name in groups):
        return nn.Sequential(OrderedDict((n, _rebuild(defaults.get(n), p)) for n, p in groups.items()))

    # nn.Sequential: indexes without weights are activations / dropout
    last = max(int(n) for n in groups)
    custom = any(str(i) in groups and (defaults.get(str(i)) is None or
                 _shapes(defaults[str(i)].state_dict()) != _shapes(groups[str(i)]))
                 for i in range(last))  # layers before the output layer changed -> your own head
    layers, need_activation = [], False
    for i in range(last + 1):
        if str(i) in groups:
            layers.append(_rebuild(defaults.get(str(i)), groups[str(i)]))
            need_activation = isinstance(layers[-1], nn.Linear)
        elif not custom and str(i) in defaults:
            layers.append(defaults[str(i)])
        elif need_activation:
            layers.append(nn.ReLU())
            need_activation = False
        else:
            layers.append(nn.Identity())
    return nn.Sequential(*layers)


def _leaf(p) -> nn.Module:
    w = p["weight"]
    if w.dim() == 2:
        return nn.Linear(w.shape[1], w.shape[0], bias="bias" in p)
    if "running_mean" in p:
        return nn.BatchNorm1d(w.shape[0])
    if w.dim() == 1:
        return nn.LayerNorm(w.shape[0])
    raise ValueError(f"Unknown layer in the classifier (weight shape {tuple(w.shape)})")


def _shapes(state) -> dict:
    return {k: tuple(v.shape) for k, v in state.items()}


def _count_outputs(model: nn.Module, size: int) -> int:
    with torch.no_grad():
        out = model(torch.zeros(1, 3, size, size))
    return _logits(out).shape[1]


def _class_names(names, n: int, lm: LoadedModel) -> list:
    for candidate in (names, config.CLASS_NAMES):
        if candidate and len(candidate) == n:
            return list(candidate)
    if config.CLASS_NAMES:
        lm.notes.append(f"This model has {n} classes but CLASS_NAMES in config.py has "
                        f"{len(config.CLASS_NAMES)} — showing class numbers instead.")
    return [f"Class {i}" for i in range(n)]


def _pretty(arch: str) -> str:
    """'mobilenet_v3_large' -> 'MobileNetV3-Large', 'vit_b_16' -> 'ViT-B/16', 'resnet18' -> 'ResNet18'."""
    if arch in PRETTY:
        return PRETTY[arch]
    for prefix, label in (("resnet", "ResNet"), ("efficientnet_", "EfficientNet-"),
                          ("convnext_", "ConvNeXt-"), ("densenet", "DenseNet")):
        if arch.startswith(prefix):
            return label + arch[len(prefix):].upper()
    return arch


HINTS = {
    "Can't get attribute": "The model class was defined in your notebook. Save the weights instead: "
                           "torch.save(model.state_dict(), 'best.pt')",
    "Could not recognise": "This is your own model class: paste it into custom_models.py "
                           "and add it to CUSTOM_MODELS",
    "size mismatch": "Add the architecture in config.py → MODEL_SETTINGS, e.g. {\"arch\": \"resnet18\"}",
    "Missing key": "Add the architecture in config.py → MODEL_SETTINGS, e.g. {\"arch\": \"resnet18\"}",
}


def _explain(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"[:600]
    hint = next((h for needle, h in HINTS.items() if needle in text), None)
    return f"{text}\n\n💡 {hint}" if hint else text


# =============================================================================
#  PREDICTION
# =============================================================================
def _logits(out) -> torch.Tensor:
    if isinstance(out, (list, tuple)):
        out = out[0]
    if hasattr(out, "logits"):  # e.g. HuggingFace style output
        out = out.logits
    return out.reshape(out.shape[0], -1)


def _to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    """Same as Resize((size, size)) + ToTensor() + Normalize(MEAN, STD)."""
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(config.MEAN)[:, None, None]) / torch.tensor(config.STD)[:, None, None]


def predict_batch(lm: LoadedModel, images: list) -> tuple[np.ndarray, float]:
    """-> (probabilities, shape (n images, n classes), milliseconds spent in the model)."""
    x = torch.stack([_to_tensor(img, lm.img_size) for img in images])
    start = time.perf_counter()
    with torch.no_grad():
        logits = _logits(lm.model(x)).float()
    ms = (time.perf_counter() - start) * 1000
    if logits.min() < 0 or not torch.allclose(logits.sum(1), torch.ones(len(logits)), atol=1e-3):
        logits = torch.softmax(logits, dim=1)   # logits -> probabilities
    return logits.numpy(), ms


def predict(lm: LoadedModel, image: Image.Image) -> tuple[np.ndarray, float]:
    """One image -> (probability of every class, milliseconds)."""
    probs, ms = predict_batch(lm, [image])
    return probs[0], ms
