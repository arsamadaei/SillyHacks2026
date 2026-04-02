# Snake Bite Identification System

A web application for identifying snakes from images and providing medical treatment advice for bite victims.

## How It Works

1. User uploads a snake image through the website
2. Custom-trained AI model identifies the snake species
3. Gemini AI provides detailed medical treatment information
4. Results are displayed to help doctors provide appropriate care

## Project Structure

```
├── website/              # Frontend files
│   ├── index.html       # Main web page
│   └── script.js        # (inline in HTML)
├── server.py            # Flask backend server
├── gemini.py            # Standalone CLI tool for Gemini analysis
├── inference.py         # Model inference functions
├── model.py             # Training script
├── snake_model.pth      # Trained model weights
└── test.png             # Test image
```

## Setup & Running

### 1. Install Dependencies

```bash
pip install torch torchvision pillow flask flask-cors google-genai
```

### 2. Run the Backend Server

```bash
python server.py
```

The server will start on `http://127.0.0.1:5000`

### 3. Open the Website

Open `website/index.html` in your browser:

```bash
# On macOS
open website/index.html

# On Linux
xdg-open website/index.html

# On Windows
start website/index.html
```

Or simply double-click the `index.html` file.

### 4. Using the Application

1. Click "Upload a picture" and select a snake image
2. Click "Analyze bite"
3. Wait for the analysis (AI model + Gemini consultation)
4. View the results including:
   - Snake identification
   - Danger level
   - Symptoms
   - Treatment steps
   - First aid instructions

## Command Line Usage

You can also use `gemini.py` directly from the command line:

```bash
# Analyze the default test image
python gemini.py

# Analyze a specific image
python gemini.py path/to/snake_image.jpg
```

## API Key

The current implementation includes a hardcoded API key for demonstration. In production, you should:

1. Set the `GEMINI_API_KEY` environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

2. Or modify the code to read from a config file

## Troubleshooting

### "Failed to fetch" error
- Make sure the server is running (`python server.py`)
- Check that the server is on port 5000
- Check browser console for more details

### Model loading issues
- Ensure `snake_model.pth` exists in the project root
- Check that PyTorch is installed correctly

### CORS errors
- The server already has CORS enabled
- If issues persist, try accessing via `http://127.0.0.1:5000` instead of `localhost`

## Disclaimer

This tool is for **informational purposes only** and is not a substitute for professional medical care. Always seek emergency medical attention for any snake bite.
