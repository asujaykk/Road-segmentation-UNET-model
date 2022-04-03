
# -*- coding: utf-8 -*-

from tensorflow import keras
import tensorflow as tf
import os


import onnxruntime
import tf2onnx
import onnxruntime as rt

model_in_size = (None, 160, 160, 3)
in_model_h5_path="models/pretrained_models/road_segmentation_160_160.h5"
out_saved_model_path="models/saved_model/road_seg
output_onnx_path = "onnx_models/road_seg.onnx"
num_classes = 3


model = keras.models.load_model(in_model_h5_path)
model.save(out_saved_model_path)
#model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

spec = (tf.TensorSpec(model_in_size, tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_onnx_path)
   
