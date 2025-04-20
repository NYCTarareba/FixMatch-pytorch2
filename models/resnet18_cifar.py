import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def build_resnet18_cifar(num_classes=10):
    """
    ResNet-18 for CIFAR-10 with modified first conv layer and removed maxpool
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove 7x7 -> 3x3 downsampling
    model.fc = nn.Linear(512, num_classes)
    return model
