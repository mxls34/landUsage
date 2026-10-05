# 🛰️ Land Usage Classifier

A Streamlit website with 3 pages (nav bar on the left):

| Page | What it does |
|---|---|
| 🔍 Predict | Press the upload button, choose **one picture** from your device, press **Predict** — every model classifies it: final answer (majority vote), each model's top-3 classes, probability of every class |
| 📊 Dashboard | Tests every model on labelled images in `test_data/<class name>/` (or an uploaded .zip): leaderboard, accuracy / speed charts, F1 per class, confusion matrix |
| 🤖 Models | The models in `models/`: architecture, classes, size |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Add your models

Put the `.pt` / `.pth` files in the `models/` folder — that's all:

```
models/best(nuum).pt
models/best_model(mix).pt
models/best(ice).pt
```

Every file there is loaded automatically. It works with any of these ways of saving:

| How the model was saved | Example |
|---|---|
| Only the weights | `torch.save(model.state_dict(), "best.pt")` |
| Whole model | `torch.save(model, "best.pt")` |
| Checkpoint | `torch.save({"model_state_dict": model.state_dict(), "class_names": classes}, "best.pt")` |
| TorchScript | `torch.jit.script(model).save("best.pt")` |

For "only the weights" the architecture is found from the weights: ResNet18/34/50,
MobileNetV3 (large/small), MobileNetV2, EfficientNet, ConvNeXt, DenseNet, ViT-B/16 …
including your own classifier head (e.g. `nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Dropout(), nn.Linear(...))`).

## Your own model class

If a model is not a plain torchvision model (like `best(nuum).pt`: MobileNetV3 features +
a projection layer), it is built from the classes in `custom_models.py`. If its predictions
look wrong, paste the real class from your training notebook there.

## Settings (`config.py`)

* `CLASS_NAMES` — your classes in training order (`ImageFolder` sorts folder names A→Z).
  Not needed if the checkpoint saved `class_names` / `classes` / `class_to_idx`.
* `IMG_SIZE`, `MEAN`, `STD` — must match the transforms used in training
  (default: `Resize((224, 224))` + ImageNet normalisation).
* `MODEL_SETTINGS` — only if one model needs something special, e.g.
  `"best(ice).pt": {"arch": "resnet18"}`.

## Big model files

GitHub refuses files over 100 MB (ViT-B/16 is ~330 MB). This repo stores `.pt` / `.pth`
with **Git LFS** (see `.gitattributes`), so install it once before adding the models:

```bash
git lfs install
git add models/*.pt
git commit -m "Add models"
git push
```
