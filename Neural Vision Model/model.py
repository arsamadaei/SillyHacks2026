"""
Load an AI model for transfer-learning and fine-tuning
Using model ResNet18
"""

import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import random


data_transforms =  transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomAdjustSharpness(.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation([x for _ in])
])
