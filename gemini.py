from inference import load_trained_model, predict
from google import genai
import sys
import os

# Configure with your API key
# NOTE: In production, use environment variables for API keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDwYPXV1vkrJIyiMqdhGlOvkFQNnutj1aY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Load your snake model once
snake_model, species_list = load_trained_model("snake_model.pth")


def analyze_snake_image(image_path):
    """
    Analyze a snake image and get treatment information from Gemini.
    
    Args:
        image_path: Path to the snake image file
        
    Returns:
        Dictionary with species, confidence, and Gemini's treatment advice
    """
    # Validate image path
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    
    # Run inference on the provided image
    species, confidence = predict(image_path, snake_model, species_list)
    
    # Create a comprehensive prompt for medical advice
    prompt = f"""You are a medical expert specializing in venomous snake bites.

A snake has been identified as "{species}" with {confidence:.1f}% confidence.

Provide detailed information for medical professionals treating a bite victim.

Include:
1. Snake common name and scientific name
2. Danger level (High/Medium/Low)
3. Description of the snake
4. Venom type and effects
5. Symptoms to expect
6. Treatment steps
7. Immediate first aid

Format as clear, actionable medical guidance."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return {
        "species": species,
        "confidence": confidence,
        "medical_advice": response.text
    }


if __name__ == "__main__":
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image for testing
        image_path = "test.png"
    
    print(f"Analyzing image: {image_path}")
    print("-" * 50)
    
    result = analyze_snake_image(image_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Snake identified: {result['species']}")

        print("\n" + "=" * 50)
        print("MEDICAL ADVICE FROM GEMINI:")
        print("=" * 50)
        print(result['medical_advice'])
