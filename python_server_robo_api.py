from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
from roboflow import Roboflow
from PIL import Image
import dotenv
import os

dotenv.load_dotenv()
API_KEY = os.getenv("ROBOFLOW_DOG_BREED_API")

app = Flask(__name__)

rf = Roboflow(API_KEY)
project = rf.workspace().project("dog_breed_simple")
model = project.version(1).model

threshold = 0.01

def preprocess_image(image):
  image = image.resize((256, 256))
  temp_image_path = "/tmp/temp_image.jpg"
  image.save(temp_image_path)
  return temp_image_path

def filter_prediction(prediction):
  # prediction decoding, reformating, filtering
  decoded_predictions = dict(prediction.json()['predictions'][0]['predictions'])
  filtered_predictions = [pred for pred in decoded_predictions.items() if float(pred[1]["confidence"]) >= threshold]
  
  # result 기본 snippet
  result = {"classification_number" : len(filtered_predictions), # 섞인 종 갯수
            "Information" : "양육정보를 추가해 주세요!"} # 양육정보 
  
  # result에 rkr prediction 아이템 추가 
  for idx, prediction_item in enumerate(filtered_predictions):
    label = prediction_item[0]
    score = prediction_item[1]["confidence"]
    result[f"classification_{idx+1:02}"] = [{"name" : label, "value" : score}] # 종 이름, 비율
  return result 

@app.route('/predict', methods=['POST'])
def predict():
  try:
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
      return jsonify({"error": "Image URL is required"}), 400
    response = requests.get(image_url)

    if response.status_code != 200:
      print("response: 200")
      return jsonify({"error": "Failed to download image"}), 400
    
    image = Image.open(BytesIO(response.content))
    temp_image_path = preprocess_image(image)
    prediction = model.predict(temp_image_path)
    result = filter_prediction(prediction)
    return result

  except Exception as e:
    return jsonify({"error": e}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5500, debug = True)