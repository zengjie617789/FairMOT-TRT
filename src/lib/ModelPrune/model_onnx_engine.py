import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)

def build_engine(model_path, engine_path):
    with trt.Builder(logger) as builder, builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = 1
        with open(model_path, 'rb') as model:
            parser.parse(model.read())
            print('Completed parse the onnx model')
        print(f'Building an engine {model_path} file, this may take a while ...')
        print(f"network num_layers: {network.num_layers}")
        # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        engine = builder.build_cuda_engine(network)
        print('Completed Creating the engine file')
        with open(engine_path, 'wb') as file:
            file.write(engine.serialize())

# def do_inference():



model_path = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34_DCN_TRT.onnx'
engine_path = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34_DCN_TRT_sample.engine'

# model_path = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34.onnx'
# engine_path = '/home/zzzj/Projects/models/onnx_engine/fairmot_dla34_sample.engine'

build_engine(model_path, engine_path)
# builder = trt.Builder(logger)
#
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#
# parser = trt.OnnxParser(network, logger)
#
# success = parser.parse_from_file(model_path)
#
# config = builder.create_builder_config()
# config.max_workspace_size = 1 << 30
# print(dir(builder))
# # serialized_engine = builder.build_serialized_network(network, config)
# serialized_engine = builder.build_cuda_engine(network, config)
#
#
# with open('sample.engine', 'wb') as f:
#     f.write(serialized_engine)
#     print('end to writing !')
