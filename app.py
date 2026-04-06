from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import base64
from io import BytesIO

import requests

OPENROUTER_API_KEY = "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"

def generate_recommendation(disease):
    prompt = f"""
    A patient is diagnosed with {disease} from an OCT scan.
    Give a short, clear medical recommendation in simple language (2-3 lines).
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    result = response.json()
    return result['choices'][0]['message']['content']

# advice = {
#     "CNV": "Consult a retina specialist immediately. Anti-VEGF injections may be required.",
#     "DME": "Maintain blood sugar control and visit an ophthalmologist. Treatment may include injections or laser therapy.",
#     "DRUSEN": "Regular eye check-ups recommended. Maintain a healthy diet rich in antioxidants.",
#     "NORMAL": "No disease detected. Maintain healthy lifestyle and regular eye checkups."
# }

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("trained_model.h5", compile=False)

class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def preprocess_image(img):
    img = img.resize((224,224))
    img = np.array(img)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No file selected")

    try:
        img = Image.open(file).convert('RGB')

        # 👉 Convert image to base64 (for preview after reload)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        processed = preprocess_image(img)
        prediction = model.predict(processed)

        predicted_class = class_names[np.argmax(prediction[0])]
        try:
            recommendation = generate_recommendation(predicted_class)
        except:
            recommendation = "AI recommendation unavailable. Please consult a doctor."

        return render_template(
            'index.html',
            prediction=predicted_class,
            recommendation=recommendation,
            image_data=img_str
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)