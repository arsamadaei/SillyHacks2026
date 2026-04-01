import google.generativeai as genai
from inference import load_trained_model, predict

# Configure with your API key
genai.configure(api_key="AIzaSyBfypU6QJLeaw3lfNSr0V8_4FiB5ODVBhw")

# Load your snake model once
snake_model, species_list = load_trained_model("snake_model.pth")

# Run inference on your image
species, confidence = predict("test.png", snake_model, species_list)
print(species)

# Gemini model
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


response = gemini_model.generate_content(
    f"Imagine you're the doctor. Your patient just got bit by a {species} snake. Tell them what a doctor would do about the symptoms, and how to treat."
)

print(f"Snake identified: {species}")
print(response.text)