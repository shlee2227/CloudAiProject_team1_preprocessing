# tensorflow == 2.15.0
import csv
from io import BytesIO
from PIL import Image
import numpy as np

from flask import Flask, request, jsonify
import requests

import tensorflow as tf
from keras.preprocessing import image


app = Flask(__name__)

# 커스텀 layer CustomDataAugmentation 정의
class CustomDataAugmentation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomDataAugmentation, self).__init__(**kwargs)
        self.flip = tf.keras.layers.RandomFlip('horizontal')
        self.rotate = tf.keras.layers.RandomRotation(0.2)

    def call(self, inputs, training = None):
        if training:
            inputs = self.flip(inputs)
            inputs = self.rotate(inputs)
        return inputs

    def get_config(self):
        config = super(CustomDataAugmentation, self).get_config()
        return config
    
# 커스텀 layer precess_input_lambda 정의를 위한 import 
preprocess_input = tf.keras.applications.vgg16.preprocess_input
# 커스텀 layer precess_input_lambda 정의 
@tf.keras.utils.register_keras_serializable()
def preprocess_input_lambda(x):
    x = preprocess_input(x)
    return x

# 모델 Load
def custom_load_model():
  tf.keras.utils.register_keras_serializable()(CustomDataAugmentation)
  # 모델 불러오기 
  model_path = "/Users/leeshinhee/Documents/newdeal/project/CloudAiProject_team1_preprocessing/models/model_tf2150_epoch200.keras"
  model = tf.keras.models.load_model(model_path, custom_objects={
      'preprocess_input_lambda': preprocess_input_lambda,
      'CustomDataAugmentation': CustomDataAugmentation,
  }, safe_mode = False, compile=False)
  # 모델 불러올때 컴파일 하지 않았으므로 여기서 컴파일 (모델 load시 컴파일 하면 에러 뜸)
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
  return model 

# verify URL
def verify_image(image_url):
  if not image_url:
    return jsonify({"error": "Image URL is required"}), 400

  response = requests.get(image_url)
  if response.status_code != 200:
    return jsonify({"error": "Failed to download image"}), 400

  image = Image.open(BytesIO(response.content))
  if image.format not in ['JPEG', 'PNG']:
    return jsonify({"error": "Unsupported image format"}), 400
  return image 

# 이미지 전처리 함수 정의
def preprocess_image(img):
    img = img.resize((224, 224))  # 이미지 로드 및 크기 조정
    img_array = image.img_to_array(img)  # 이미지를 numpy 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = preprocess_input(img_array)  # 모델에 맞게 전처리
    return img_array

# Predictions 후처리
def filtered_predictions(predictions):
  predictions = predictions[0].tolist() 
  formated_predictions = [{"id": id, "score": float(score)} for id, score in enumerate(predictions)]
  filtered_predictions = [pred for pred in formated_predictions if pred["score"] >= threshold]

  # result 기본 snippet
  result = {"classification_number" : len(filtered_predictions), # 섞인 종 갯수
            "Information" : "양육정보를 추가해주세요!"} # 양육정보 
  
  # result에 rkr prediction 아이템 추가 
  for idx, prediction_item in enumerate(filtered_predictions):
    label = labels[prediction_item["id"]] if prediction_item["id"] in labels else prediction_item["id"]
    score = round(prediction_item["score"], 2)
    result[f"classification_{idx+1:02}"] = [{"name" : label, "value" : score}] # 종 이름, 비율
  return result    

threshold = 0.01
BATCH_SIZE =16
IMG_SIZE = (224, 224)
model = custom_load_model()

labels_path = "/Users/leeshinhee/Documents/newdeal/project/CloudAiProject_team1_preprocessing/models/labels.csv"
with open(labels_path, mode='r', encoding='UTF-8') as f:
  csv_reader = csv.reader(f)
  labels = {}
  for item in csv_reader:
    labels[int(item[0])] = item[1].strip()

@app.route('/predict', methods=['POST'])
def predict():
  try:
    data = request.json
    image_url = data.get('image_url')
    img = verify_image(image_url) # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    img_arr = preprocess_image(img) # <class 'numpy.ndarray'>
    predictions = model.predict(img_arr) 
    results = filtered_predictions(predictions)
    return jsonify(results)

  except Exception as e:
    return jsonify({"error": e}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5500, debug=True)


