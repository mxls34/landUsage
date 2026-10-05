# 🛰️ Land Usage Classifier

A Streamlit website: upload **one satellite image**, press **Predict**, and every model in
`models/` classifies the land usage. You see the final answer (majority vote), each model's
answer with its top-3 classes, and the probability of every class.

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
