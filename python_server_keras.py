from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

def preprocess_image(image):
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image

def decode_predictions(predictions):
    return tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        image = Image.open(BytesIO(response.content))
        if image.format not in ['JPEG', 'PNG']:
            return jsonify({"error": "Unsupported image format"}), 400

        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions)

        results = [{"label": label, "score": float(score)} for _, label, score in decoded_predictions]
        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)