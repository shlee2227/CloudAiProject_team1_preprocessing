# tensorflow == 2.15.0
import os
from flask import Flask, request, jsonify
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)

threshold = 0.01

dir_to_test = '/Users/leeshinhee/Documents/newdeal/project/code_models/dataset/dataset2_70/5_dog_breed_image_dataset/'


# 커스텀 layer 정의 
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

@tf.keras.utils.register_keras_serializable()
def preprocess_input_lambda(x):
    x = tf.keras.applications.vgg16.preprocess_input.preprocess_input(x)
    return (x,)
# 커스텀 모델 정의 

# ------------model load------------
def custom_load_model(file_path):
  tf.keras.utils.register_keras_serializable()(CustomDataAugmentation)
  # data_augmentation = CustomDataAugmentation()
  
  # 모델 불러오기 
  model_path = "/Users/leeshinhee/Documents/newdeal/project/CloudAiProject_team1_preprocessing/models/model_tf2150_epoch100_240610_1702.keras"
  model = tf.keras.models.load_model(model_path, custom_objects={
      'preprocess_input_lambda': preprocess_input_lambda,
      'CustomDataAugmentation': CustomDataAugmentation,
  }, safe_mode = False, compile=False)

  # 옵티마이저 문제가 발생하여 모델에서 불러오지 않고 재 설정
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])



# 로컬 디렉토리의 파일 테스트용
@app.route('/predictdir', methods=['POST'])
def predict_in_derectory():
  test_dir = os.path.join(file_path, 'test')

  BATCH_SIZE =16
  IMG_SIZE = (224, 224)

  test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            label_mode = 'int')
  class_names = test_dataset.class_names

  # test
  loss, accuracy = model.evaluate(test_dataset)
  print("test loss:", loss)
  print('Test accuracy :', accuracy)

  # test image 전달
  image_batch, label_batch = test_dataset.as_numpy_iterator().next()
  predictions = model.predict_on_batch(image_batch)

  # softmax 함수로 return하도록 함
  predictions_classes = tf.nn.softmax(predictions)
  predictions = tf.argmax(predictions_classes, axis = 1)

  print('Predictions per class:\n', predictions_classes.numpy())
  print('Predictions:\n', predictions.numpy())
  print('Labels:\n', label_batch)

  plt.figure(figsize=(10, 10))
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
  plt.show()


def filtered_predictions(predictions):
  # prediction decoding, reformating, filtering
  decoded_predictions = tf.keras.applications.vgg16.decode_predictions(predictions, top=10)[0]
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
    
    predictions = model.predict(image)
    results = filtered_predictions(predictions)

    return jsonify(results)

  except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5500, debug=True)


