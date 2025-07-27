from flask import Flask, request, jsonify, render_template
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

app = Flask(__name__)

# Load the BLIP model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Route for the main HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle image upload and generate caption
@app.route("/caption", methods=["POST"])
def caption_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=7860)
