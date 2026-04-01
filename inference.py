import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['classes']
    num_species = len(class_names)

    model = models.efficientnet_b0() 
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_species)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names

inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
    MODEL_PATH = "snake_model.pth"
    TEST_IMAGE = "test.png" 
    
    if os.path.exists(MODEL_PATH):
        print("Loading model...")
        snake_model, species_list = load_trained_model(MODEL_PATH)
        
        if os.path.exists(TEST_IMAGE):
            name, score = predict(TEST_IMAGE, snake_model, species_list)
            print(f"\nResult: {name}")
            print(f"Confidence: {score:.2f}%")
        else:
            print(f"Error: Could not find image '{TEST_IMAGE}'")
    else:
        print("Error: No trained model file found. Run your training script first!")