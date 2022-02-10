import onnx
import sys

file_name = '/home/awifi/zzzj/Projects/FairMOT/models/ctdet_coco_dla_2x.onnx'
model = onnx.load(file_name)
onnx.checker.check_model(model)