from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import torch
import os
from PIL import Image
import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration

# Flask app setup
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Gemini API (Use your API key)
API_KEY = "AIzaSyA4hlHSzgK1-nyBURZ8XD5dvNjimYG-j8A"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Define topic for Q&A generation
TOPIC = "power generation concept in class 9th"

# Function to get available Gemini model
def get_gemini_model():
    """Check available models and return the correct one."""
    try:
        models = genai.list_models()
        available_models = [model.name for model in models]
        print("Available Gemini models:", available_models)  # Debugging line

        # Prioritize Gemini 1.5 or 2.0 models
        for model in [
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-pro-002",
            "models/gemini-2.0-pro-exp",
        ]:
            if model in available_models:
                return model

        raise ValueError("No suitable Gemini model found. Check API key and access.")
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return None  # Avoid using an invalid model


# Get the correct Gemini model
GEMINI_MODEL = get_gemini_model()

def extract_keyframes(video_path, interval=30):
    """Extract key frames from a video at a given interval."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def generate_caption(image):
    """Generates a description for an image using BLIP."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    
    return caption

def merge_captions(captions):
    """Merge multiple captions into a meaningful summary using Gemini API."""
    if not captions:
        return "No captions generated."

    if not GEMINI_MODEL:
        return "⚠️ No valid Gemini model found. Cannot merge captions."

    prompt = f"Combine the following captions into a meaningful summary:\n\n{'. '.join(captions)}"
    
    try:
        gen_model = genai.GenerativeModel(GEMINI_MODEL)
        response = gen_model.generate_content(prompt)
        return response.text.strip() if response.text else "No meaningful summary generated."
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def generate_questions_answers(final_caption):
    """Generate questions and answers using Gemini API."""
    if not final_caption.strip():
        return "⚠️ No caption available to generate questions."

    if not GEMINI_MODEL:
        return "⚠️ No valid Gemini model found. Cannot generate Q&A."

    prompt = f"Based on the following description: '{final_caption}', generate 3 questions and their answers related to {TOPIC}."
    
    try:
        gen_model = genai.GenerativeModel(GEMINI_MODEL)
        response = gen_model.generate_content(prompt)
        return response.text.strip() if response.text else "⚠️ No questions generated."
    except Exception as e:
        return f"Error generating questions: {str(e)}"

@app.route('/')
def index():
    return render_template('video.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handles file upload and processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    captions = []

    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):  # If video
        frames = extract_keyframes(filepath)
        captions = [generate_caption(frame) for frame in frames]
    else:  # If image
        image = cv2.imread(filepath)
        captions.append(generate_caption(image))

    final_caption = merge_captions(captions)
    qna = generate_questions_answers(final_caption)

    os.remove(filepath)  # Clean up uploaded file

    return jsonify({
        "final_caption": final_caption,
        "qna": qna
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Use PORT from Render, default to 10000
    app.run(host="0.0.0.0", port=port, debug=True)
