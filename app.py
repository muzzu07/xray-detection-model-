import os
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# ✅ Load trained model
MODEL_PATH = 'best_model.keras'

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

CLASS_NAMES = ['COVID-19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']


# ===============================
# 🏥 Disease Knowledge Base
# ===============================
disease_info = {
    "NORMAL": {
        "title": "Normal Lung Radiograph",
        "base_description": "No significant radiographic abnormalities detected in the lung fields.",
        "clinical_note": "Lung parenchyma appears clear without focal consolidation, cavitation, or suspicious opacity.",
        "risk": "Low"
    },
    "PNEUMONIA": {
        "title": "Pneumonia Detected",
        "base_description": "Radiographic findings suggest inflammatory changes within the lung fields.",
        "clinical_note": "Possible presence of localized consolidation or patchy opacities consistent with pneumonia.",
        "risk": "Moderate"
    },
    "TUBERCULOSIS": {
        "title": "Pulmonary Tuberculosis Pattern Detected",
        "base_description": "Imaging features may indicate tuberculosis-related lung involvement.",
        "clinical_note": "Upper lobe infiltrates or cavitary lesions may be present, commonly associated with TB.",
        "risk": "High"
    },
    "COVID-19": {
        "title": "COVID-19 Associated Pattern Detected",
        "base_description": "Findings may correspond to viral pneumonia patterns.",
        "clinical_note": "Ground-glass opacities or bilateral peripheral involvement may be observed.",
        "risk": "Moderate"
    }
}


# ===============================
# 🖼 Image Preprocessing
# ===============================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Same preprocessing as training
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ===============================
# 🌐 Routes
# ===============================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    # 🔍 Model Prediction
    predictions = model.predict(processed_image, verbose=0)[0]

    class_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[class_index]
    confidence = float(predictions[class_index] * 100)

    # 🔎 Get disease info
    info = disease_info[predicted_class]

    # ===============================
    # 🎯 Confidence-Based Logic
    # ===============================
    if confidence >= 90:
        certainty = "High diagnostic confidence based on AI pattern recognition."
        recommendation = "Immediate clinical evaluation is strongly recommended."
    elif confidence >= 75:
        certainty = "Moderate diagnostic confidence."
        recommendation = "Further clinical correlation and confirmatory testing are advised."
    else:
        certainty = "Low diagnostic certainty. Results should be interpreted with caution."
        recommendation = "Additional imaging or laboratory testing is recommended before confirmation."

    # 📝 Final Structured Description
    final_description = f"""
{info['base_description']}

Clinical Interpretation:
{info['clinical_note']}

Diagnostic Certainty:
{certainty}
"""

    return jsonify({
        "prediction": predicted_class,
        "title": info["title"],
        "description": final_description.strip(),
        "recommendation": recommendation,
        "risk": info["risk"],
        "confidence": round(confidence, 2)
    })


# ===============================
# 🚀 Run App
# ===============================
if __name__ == '__main__':
    app.run(debug=True)