"""
Load an AI model for transfer-learning and fine-tuning
Using model EfficientNet-B0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import glob

import os
from tqdm import tqdm
import random

print(os.path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights = 'IMAGENET1K_V1') # A good model for recognizing textures which is useful for snakes
checkpoint_path = "snake_model_tweaking.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 20

print(f"Using device {device}.")

data_transforms =  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomAdjustSharpness(.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder('Snakes_Dataset', data_transforms)
train_loader = DataLoader(train_set, BATCH_SIZE, True)

num_species = len(train_set.classes)

# Replace the previous connected layers to the new connected layer with num_species amound of nodes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_species)

if os.path.exists(checkpoint_path):
    print(f"Found saved model '{checkpoint_path}'. Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Unfreeze all layers for post-training
    for param in model.parameters():
        param.requires_grad = True

    print("Model unfrozen. Fine-tuning all layers...")

else:
    print("No saved model found. Starting training from scratch.")
    
model = model.to(device)



# Gradient descent by minimizing cross-entropy loss, using adam optimizer
loss = nn.CrossEntropyLoss()
print(loss)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

print(f"Model ready to classify {num_species} snake species.")


def run_epoch_samples(model, class_names, num_samples=10):
    model.eval()
    all_images = glob.glob("Snakes_Dataset/**/*.*", recursive=True)
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    print("\n--- Model Check (Random Samples) ---")
    for path in samples:
        true_label = os.path.basename(os.path.dirname(path))
        img = Image.open(path).convert("RGB")
        img_t = data_transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_t)
            _, pred = torch.max(output, 1)
            predicted_label = class_names[pred.item()]
        
        status = "correct" if true_label == predicted_label else "incorrect"
        print(f"{status} True: {true_label} | AI: {predicted_label}")
    print("------------------------------------\n")


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        current_loss = loss(outputs, labels)
        current_loss.backward()
        optimizer.step()
        
        running_loss += current_loss.item()

        current_avg_loss = running_loss / (progress_bar.n + 1)
        progress_bar.set_postfix(loss=f"{current_avg_loss:.4f}")
    
    run_epoch_samples(model, train_set.classes)
    
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")
    torch.save({'model_state_dict': model.state_dict(), 'classes': train_set.classes}, "snake_model_tweaking.pth")


print("Training Complete!")
