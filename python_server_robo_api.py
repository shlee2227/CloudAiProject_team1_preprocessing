from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
from roboflow import Roboflow
from PIL import Image
import dotenv
import os

dotenv.load_dotenv()
API_KEY = os.getenv("ROBOFLOW_REF_MODEL_API")

app = Flask(__name__)

rf = Roboflow(API_KEY)
project = rf.workspace().project("dog_breed_simple")
model = project.version(1).model

def preprocess_image(image):
  image = image.resize((256, 256))
  temp_image_path = "/tmp/temp_image.jpg"
  image.save(temp_image_path)
  return temp_image_path

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
    # print(response.json)
    image = Image.open(BytesIO(response.content))
    temp_image_path = preprocess_image(image)
    predictions = model.predict(temp_image_path)
    return predictions.json()

  except Exception as e:
    return jsonify({"error": e}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5500)