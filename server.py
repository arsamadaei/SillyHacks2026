"""
Flask backend server for snake bite identification.
Receives image from frontend, runs AI model inference, queries Gemini for treatment info.
"""

import os
import base64
import json
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from inference import load_trained_model, predict
from google import genai

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = "snake_model.pth"
UPLOAD_FOLDER = "uploads"

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the snake model once at startup
print("Loading snake model...")
snake_model, species_list = load_trained_model(MODEL_PATH)
print(f"Model loaded. Can identify {len(species_list)} snake species.")

# Configure Gemini API
# NOTE: In production, use environment variables for API keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDwYPXV1vkrJIyiMqdhGlOvkFQNnutj1aY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def get_snake_info_from_gemini(species, confidence):
    """
    Query Gemini for detailed snake information and treatment advice.
    """
    prompt = f"""You are a medical expert specializing in venomous snake bites.

A snake has been identified as "{species}" with {confidence:.1f}% confidence.

Provide detailed information for medical professionals treating a bite victim.

Return ONLY a valid JSON object (no markdown, no code fences) with this exact structure:

{{
  "snakeName": "Common name of the snake",
  "scientificName": "Scientific/Latin name",
  "dangerLevel": "High" or "Medium" or "Low",
  "description": "Brief description of the snake's appearance, habitat, and behavior",
  "venomType": "Type of venom (e.g., neurotoxic, hemotoxic, cytotoxic) and how it affects the body",
  "symptoms": ["symptom 1", "symptom 2", "symptom 3", "symptom 4", "symptom 5"],
  "treatment": ["treatment step 1", "treatment step 2", "treatment step 3", "treatment step 4"],
  "firstAid": "Immediate first aid steps before reaching hospital. Be specific and medically accurate.",
  "confidence": {int(confidence)}
}}

Be medically accurate and thorough. If the species is unknown or not a venomous snake, indicate that in the dangerLevel and provide appropriate guidance."""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        result = json.loads(response_text)
        return result
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Return fallback response if Gemini fails
        return {
            "snakeName": species,
            "scientificName": "",
            "dangerLevel": "Unknown",
            "description": f"Snake identified as {species}.",
            "venomType": "Unknown - consult local toxicology reference",
            "symptoms": ["Consult medical reference for species-specific symptoms"],
            "treatment": ["Seek immediate medical attention", "Contact local poison control center"],
            "firstAid": "Keep victim calm and still. Immobilize the affected limb. Seek emergency medical care immediately.",
            "confidence": int(confidence)
        }


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main endpoint for snake bite image analysis.
    Expects JSON with: { "image_data": "base64_encoded_image", "mime_type": "image/jpeg" }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({"error": {"message": "No image data provided"}}), 400
        
        # Decode base64 image
        image_data = data['image_data']
        mime_type = data.get('mime_type', 'image/jpeg')
        
        # Determine file extension from mime type
        ext = 'jpg'
        if 'png' in mime_type:
            ext = 'png'
        elif 'gif' in mime_type:
            ext = 'gif'
        elif 'webp' in mime_type:
            ext = 'webp'
        
        # Save the image temporarily
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_upload.{ext}")
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image.save(temp_path)
        
        print(f"Image saved to {temp_path}")
        
        # Run model inference
        species, confidence = predict(temp_path, snake_model, species_list)
        print(f"Prediction: {species} ({confidence:.2f}% confidence)")
        
        # Get detailed info from Gemini
        result = get_snake_info_from_gemini(species, confidence)
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": {"message": str(e)}}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": True})


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with instructions."""
    return """
    <h1>Snake Bite Identification API Server</h1>
    <p>The server is running correctly!</p>
    <p><strong>To use the application:</strong></p>
    <ol>
        <li>Open <code>website/index.html</code> directly in your browser (double-click the file)</li>
        <li>Upload a snake image and click "Analyze bite"</li>
    </ol>
    <p>Available endpoints:</p>
    <ul>
        <li><code>POST /analyze</code> - Analyze snake image</li>
        <li><code>GET /health</code> - Health check</li>
    </ul>
    """


if __name__ == '__main__':
    print("Starting server on http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    app.run(host='127.0.0.1', port=5000, debug=True)
