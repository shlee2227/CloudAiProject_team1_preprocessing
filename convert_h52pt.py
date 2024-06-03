# h5 to onnx
import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx

# Load your .h5 model
tf_model = tf.saved_model.load('/Users/leeshinhee/Documents/newdeal/project/CloudAiProject_team1_preprocessing/saved_model.pb')

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(tf_model, output_path='model.onnx')

# onnx to pt
import onnx
import onnx2pytorch

# Load the ONNX model
onnx_model = onnx.load('model.onnx')

# Convert ONNX model to PyTorch
pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

# Save the PyTorch model
import torch
torch.save(pytorch_model.state_dict(), 'model.pt')


# use pt
from ultralytics import YOLO

custom_weights = "model.pt"
model = YOLO(f"{custom_weights}")

def predict(path):
    result = model.predict(source=path, save=False, stream=True,
                           conf=0.6, iou=0.01, imgsz=320, vid_stride=6)
    return result
