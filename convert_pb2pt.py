import tensorflow as tf
import onnx
import torch
import tf2onnx
import onnxruntime
from onnxruntime import GraphOptimizationLevel, SessionOptions

# Load the TensorFlow model
tf_model = tf.saved_model.load('saved_model.pb')

# Convert the TensorFlow model to ONNX
onnx_model, _ = tf2onnx.convert.from_saved_model('saved_model.pb', opset=13)

# Save the ONNX model
onnx.save(onnx_model, 'model.onnx')

# Create session options
session_options = SessionOptions()
session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

# Load the ONNX model
onnx_model = onnx.load('model.onnx')

# Convert ONNX model to PyTorch
torch_model = onnxruntime.training.ortmodule.convert(onnx_model, session_options)

# Save the PyTorch model
torch.save(torch_model._state_dict, 'model.pt')

