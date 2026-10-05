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


class ResNet18Attention(nn.Module):
    """best.pt (ice) — ResNet18 features -> 1x1 conv 512->128 -> self-attention over the
    7x7 positions (+ gated residual, LayerNorm) -> average -> Linear.

    ⚠️ Rebuilt from the saved weights only. The number of attention heads and how
    "gate" is used cannot be read from the weights — if the predictions look wrong,
    replace this class with the one from your training notebook.
    """
    arch_name = "ResNet18 + attention"

    def __init__(self, num_classes: int, dim: int = 128, heads: int = 4):
        super().__init__()
        r = tvm.resnet18()
        # backbone.0 conv1, .1 bn1, .2 relu, .3 maxpool, .4-.7 layer1-4
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.projection = nn.Sequential(nn.Conv2d(512, dim, 1), nn.ReLU(inplace=True))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.projection(self.backbone(x))            # (B, 128, 7, 7)
        tokens = x.flatten(2).transpose(1, 2)              # (B, 49, 128)
        attended, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm(tokens + torch.tanh(self.gate) * attended)
        return self.classifier(tokens.mean(dim=1))


# Tried in this order
CUSTOM_MODELS = [
    MobileNetV3Projection,
    ResNet18Attention,
]
