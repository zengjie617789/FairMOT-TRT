import torch
import tensorrt as trt

print(f"torch version: {torch.__version__}, \n"
      f"cuda version: {torch.version.cuda} \n"
      f"tensorrt version: {trt.__version__} \n"
      f"cudnn version: {torch.backends.cudnn.version()}")
