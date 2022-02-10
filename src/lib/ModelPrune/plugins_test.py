# from torch2trt.plugins import GroupNormPlugin
# from torch2trt.torch2trt import InterpolatePlugin
# import torch2trt
import tensorrt as trt
import torch
# PLUGIN_NAME = 'interpolate'
# registry = trt.get_plugin_registry()
# for c in registry.plugin_creator_list:
#     print("plugin name:", c.name, "plugin namespace:", c.plugin_namespace)
#     if c.name == PLUGIN_NAME and c.plugin_namespace == 'torch2trt':
#         creator = c
print(torch.version.cuda)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
    print(plugin_creator.name)
    if plugin_creator.name == 'DCNv2_TRT':
    # if plugin_creator.name == 'MultilevelCropAndResize':
        dcnCreator = plugin_creator
        print("!!!!!")
