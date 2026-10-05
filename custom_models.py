"""
Your own model classes (models that are not a plain torchvision model).

When a .pt file only has weights and is not a normal ResNet / MobileNet / ViT ...,
the website tries every class in CUSTOM_MODELS below. The first one whose weights
fit is used. To add one: paste the class from your training notebook and add it
to CUSTOM_MODELS. Its __init__ must accept num_classes.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class MobileNetV3Projection(nn.Module):
    """best(nuum).pt — MobileNetV3-Large features -> 1x1 conv 960->128 -> pool -> Linear.

    ⚠️ Rebuilt from the saved weights only. Weights cannot show the layers without
    weights (ReLU, Dropout, pooling), so if the predictions look wrong, replace this
    class with the one from your training notebook.
    """
    arch_name = "MobileNetV3 + projection"

    def __init__(self, num_classes: int, proj_dim: int = 128):
        super().__init__()
        self.backbone = tvm.mobilenet_v3_large().features            # backbone.0 ... backbone.16
        self.projection = nn.Sequential(nn.Conv2d(960, proj_dim, 1), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(proj_dim, num_classes))

    def forward(self, x):
        x = self.projection(self.backbone(x))
        return self.classifier(torch.flatten(self.pool(x), 1))


# Tried in this order
CUSTOM_MODELS = [
    MobileNetV3Projection,
]
