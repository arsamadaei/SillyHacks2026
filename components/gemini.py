import google.generativeai as genai
from model import load_trained_model, predict

# Configure with your API key
genai.configure(api_key="AIzaSyCe5CzEInTfVjwEjp3dOCkv_SZAoozVzM8")

# Load your snake model once
snake_model, species_list = load_trained_model("snake_model.pth")

# Run inference on your image
species, confidence = predict("test.png", snake_model, species_list)

# Gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Use the species result in your prompt
snake = species

response = gemini_model.generate_content(
    f"Imagine you're the doctor. Your patient just got bit by a {snake} snake. "
    f"Tell them what a doctor would, "
    f"about the symptoms, and how to treat."
)

print(f"Snake identified: {snake}")
print(response.text)