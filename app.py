import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Static topic
topic = "reverse power generation for class 9th"

def generate_caption(image_path, api_key):
    # Load BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Generate questions using Gemini API with static topic
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Generate 5 questions based on this description: {caption}. The topic is: {topic}")
    questions = response.text.strip() if response and response.text else "No questions generated."
    
    return {"description": caption, "questions": questions}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Replace with your Gemini API key
    api_key = "AIzaSyAtZdcm9nN--eMNlWoiF0wRuTwE70mBkV4"
    result = generate_caption(filepath, api_key)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
