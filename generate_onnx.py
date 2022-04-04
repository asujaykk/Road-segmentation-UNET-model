
# -*- coding: utf-8 -*-

from tensorflow import keras
import tensorflow as tf
import os


import onnxruntime
import tf2onnx
import onnxruntime as rt
import argparse

#parse the commandline parameters

parser = argparse.ArgumentParser(description = 'Generate onnx model from keras model')
parser.add_argument('--input',metavar='string',type=str,required = True,help='Path to keras model (.h5 file)')
parser.add_argument('--output',metavar='String',type=str,default="models/onnx_models/road_seg_160_160.onnx",help='Path to output onnx model file')
parser.add_argument('--temp',metavar='Integer',type=str,default="models/saved_model/road_seg",help='Path where keras model format will be saved during processing')


args = parser.parse_args()


model_in_size = (None, 160, 160, 3)
in_model_h5_path=args.input     # "models/pretrained_models/road_segmentation_160_160.h5"
out_saved_model_path=args.temp  #     "models/saved_model/road_seg"
output_onnx_path =args.output   # "models/onnx_models/road_seg_160_160.onnx"
num_classes = 3


model = keras.models.load_model(in_model_h5_path)
model.save(out_saved_model_path)
#model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

spec = (tf.TensorSpec(model_in_size, tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_onnx_path)
   
