"""
Load an AI model for transfer-learning and fine-tuning
Using model EfficientNet-B0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

import os

print(os.path)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = models.efficientnet_b0(weights = 'IMAGENET1K_V1') # A good model for recognizing textures which is useful for snakes
BATCH_SIZE = 16

print(f"Using device {device}.")

data_transforms =  transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomAdjustSharpness(.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder('Snakes_Dataset', data_transforms)
train_loader = DataLoader(train_set, BATCH_SIZE, True)

num_species = len(train_set.classes)

# Replace the previous connected layers to the new connected layer with num_species amound of nodes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_species)

model = model.to(device)

# Gradient descent by minimizing cross-entropy loss, using adam optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

print(f"Model ready to classify {num_species} snake species.")
