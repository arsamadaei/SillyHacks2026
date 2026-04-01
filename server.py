from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import base64
from PIL import Image
import io
import json
import os

# Import your existing PyTorch logic
from inference import load_trained_model, predict

app = Flask(__name__)
# This allows your HTML file to communicate with this server
CORS(app) 

# --- 1. SETUP ---
# Remember to insert your real API key, but keep it out of public GitHub repos!
genai.configure(api_key="YOUR_GEMINI_API_KEY") 
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

print("Loading PyTorch model into memory...")
# Make sure "snake_model.pth" is in the same directory as this script
snake_model, species_list = load_trained_model("snake_model.pth")
print("Models ready! Starting server...")

# --- 2. THE API ENDPOINT ---
@app.route('/analyze', methods=['POST'])
def analyze_bite():
    # Grab the JSON data sent from index.html
    data = request.json
    image_data = data.get('image_data')

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode the base64 string back into an image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Your inference.py expects a file path, so we'll save it temporarily
        temp_filename = "temp_web_upload.png"
        image.save(temp_filename)

        # Run your PyTorch prediction
        species, confidence = predict(temp_filename, snake_model, species_list)
        
        # Clean up the temporary file so they don't pile up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        # --- 3. GET GEMINI CONTEXT (STRICT JSON MODE) ---
        prompt = f"""
        You are a medical expert specializing in venomous snake bites. 
        A patient was just bitten by a {species}.

        Return a JSON object with this exact structure:
        {{
          "snakeName": "{species}",
          "scientificName": "Provide the scientific name",
          "dangerLevel": "High, Medium, Low, or Unknown",
          "description": "2-3 sentences on the snake appearance, habitat, distribution.",
          "venomType": "Description of venom type and mechanism.",
          "symptoms": ["symptom 1", "symptom 2", "symptom 3"],
          "treatment": ["step 1", "step 2", "step 3"],
          "firstAid": "2-3 sentences of immediate first aid before reaching hospital.",
          "confidence": {confidence}
        }}
        """
        
        # Force Gemini to ONLY output JSON
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
        )

        # Convert the guaranteed JSON string to a Python dictionary
        result_json = json.loads(response.text)
        
        # Send it back to the frontend!
        return jsonify(result_json)

    except Exception as e:
        # If anything goes wrong, send the error back to the website
        # The print statement helps you see the error in your terminal
        print(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Changed from 5000 to 5001 to avoid potential AirPlay conflicts
    app.run(port=5001, debug=True)