import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="AIzaSyCe5CzEInTfVjwEjp3dOCkv_SZAoozVzM8")

# Choose a model
model = genai.GenerativeModel("gemini-2.0-flash")  # fast & cheap
# model = genai.GenerativeModel("gemini-1.5-pro")  # more capable

# Send a prompt
snake = other_ai_result  # whatever variable holds the output from your other AI

response = model.generate_content(f"Imagine you're the doctor. Your patient just got bit by a {snake} snake. Tell them what a doctor would, about the symptoms, and how to treat.")
print(response.text)