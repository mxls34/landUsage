"""
⚙️ SETTINGS — you only need this file if something doesn't look right.
Put your .pt files in the models/ folder: every file there is used automatically.
"""

# Folder with your models (.pt / .pth)
MODELS_FOLDER = "models"

# Class names, SAME ORDER as in training
CLASS_NAMES = ['Agriculture', 'Nature_Forest', 'Water', 'Residential', 'Industrial_Infrastructure']

# Emoji shown next to each class name
CLASS_ICONS = {
    "Agriculture": "🌾", "Nature_Forest": "🌲", "Water": "🌊",
    "Residential": "🏘️", "Industrial_Infrastructure": "🏭",
}

# Image pre-processing — same as the torchvision transforms used in training:
#   transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]     # ImageNet values (used by MobileNetV3, ViT_B_16, ResNet18 weights)
STD = [0.229, 0.224, 0.225]

# Optional: settings for ONE model, by file name. Usually not needed — the architecture
# and the classes are read from the file. Example:
#   "best(ice).pt": {"arch": "resnet18", "classes": ["Forest", "River", ...], "img_size": 224},
# "arch" can be: resnet18, resnet34, resnet50, mobilenet_v3_large, mobilenet_v3_small,
#                mobilenet_v2, efficientnet_b0, vit_b_16, vit_b_32, convnext_tiny, densenet121
MODEL_SETTINGS = {
}

# Dashboard: labelled test images, one sub-folder per class (folder name = class name):
#   test_data/Forest/img1.jpg, test_data/River/img2.jpg, ...   (or upload a .zip on the page)
TEST_FOLDER = "test_data"
MAX_IMAGES_PER_CLASS = None      # e.g. 100 for a faster test, None = every image
BATCH_SIZE = 16
