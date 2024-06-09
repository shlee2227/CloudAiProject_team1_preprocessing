from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
threshold = 0.01

def preprocess_image(image):
  image = image.resize((224, 224))
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = tf.keras.applications.resnet50.preprocess_input(image)
  image = tf.expand_dims(image, axis=0)
  return image


def filtered_predictions(predictions):
  # prediction decoding, reformating, filtering
  decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=10)[0]
  formated_predictions = [{"label": label, "score": float(score)} for _, label, score in decoded_predictions]
  filtered_predictions = [pred for pred in formated_predictions if pred["score"] >= threshold]

  # result 기본 snippet
  result = {"classification_number" : len(filtered_predictions), # 섞인 종 갯수
            "Information" : "양육정보를 추가해주세요!"} # 양육정보 
  
  # result에 rkr prediction 아이템 추가 
  for idx, prediction_item in enumerate(filtered_predictions):
    label = prediction_item["label"]
    score = prediction_item["score"]
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
      return jsonify({"error": "Failed to download image"}), 400

    image = Image.open(BytesIO(response.content))
    if image.format not in ['JPEG', 'PNG']:
      return jsonify({"error": "Unsupported image format"}), 400
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    results = filtered_predictions(predictions)

    return jsonify(results)

  except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5500, debug=True)


