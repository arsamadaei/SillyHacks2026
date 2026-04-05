import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import glob
import os
import random

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(checkpoint_path):
    # Load the saved dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['classes']
    num_species = len(class_names)

    # Recreate the exact EfficientNet-B0 structure
    model = models.efficientnet_b0() 
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_species)
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval() # Set to evaluation mode
    
    return model, class_names

# Use simpler transforms for checking (avoids extreme random crops that hide the snake)
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_batch_check(model, class_names, num_samples=10):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Get all image paths
    all_images = glob.glob("Snakes_Dataset/**/*.*", recursive=True)
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    running_loss = 0.0
    correct_count = 0
    
    print(f"\n--- Model Check ({num_samples} Random Samples) ---")
    
    for path in samples:
        # Get true label from folder name
        true_label_name = os.path.basename(os.path.dirname(path))
        true_label_idx = torch.tensor([class_names.index(true_label_name)]).to(device)
        
        # Prep image
        img = Image.open(path).convert("RGB")
        img_t = inference_transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_t)
            
            # Calculate Loss for this sample
            sample_loss = criterion(output, true_label_idx)
            running_loss += sample_loss.item()
            
            # Calculate Confidence & Prediction
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, pred_idx = torch.max(probabilities, 0)
            predicted_label = class_names[pred_idx.item()]
            
            # Track Accuracy
            is_correct = (predicted_label == true_label_name)
            if is_correct:
                correct_count += 1
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG  "
            print(f"{status} | True: {true_label_name:20} | AI: {predicted_label:20} ({confidence.item()*100:.1f}%)")

    # Final Batch Stats
    avg_loss = running_loss / num_samples
    accuracy = (correct_count / num_samples) * 100
    
    print("-" * 60)
    print(f"BATCH STATISTICS:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy:     {accuracy:.2f}% ({correct_count}/{num_samples})")
    print("-" * 60 + "\n")


def predict(image_path, model, class_names):
    img = Image.open(image_path).convert("RGB")
    img_t = inference_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred_idx = torch.max(probabilities, 0)

    species = class_names[pred_idx.item()]
    confidence = conf.item() * 100
    
    return species, confidence

if __name__ == "__main__":
    MODEL_PATH = "snake_model_tweaking.pth" 
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        snake_model, species_list = load_trained_model(MODEL_PATH)
    
        # Run the check
        predict_batch_check(snake_model, species_list, num_samples=10) 
    else:
        print(f"Error: Could not find '{MODEL_PATH}'. Ensure the training script has finished at least one epoch.")