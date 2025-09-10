import hls4ml
import qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.core.onnx_exec import execute_onnx
# from hls4ml.converters import convert_from_keras_model, load_saved_model
# set random seed for reproducibility
import numpy as np
np.random.seed(1998)


def main():
    # Load the ONNX model
    qonnx_path = "onnx_files/subcnn_qonnx.onnx"
    model = ModelWrapper(qonnx_path)
    model = cleanup_model(model)
    model = model.transform(ConvertToChannelsLastAndClean())
    model = cleanup_model(model)
    
    config = hls4ml.utils.config.config_from_onnx_model(
        model, granularity='name', backend='Vitis', default_precision='fixed<16,6>'
    )
    
    config['LayerName']['MaxPool_0'] = {
        'pad_left': 0,    
        'pad_right': 0  
    }
    for layer in config['LayerName']:
        if "rescale" not in layer:
            config['LayerName'][layer]['ReuseFactor'] = 4

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir='subcnn_hls4ml',
        io_type='io_stream',
        backend='Vitis',
        hls_config=config,
    )
    
    hls_model.compile()
    #hls_model.save("subcnn_hls4ml.fml")
    # load hls4ml model
    # hls_model = load_saved_model("subcnn_hls4ml.fml")

    random_input = np.random.rand(1, 1, 32).astype('float32')
    input_dict = {'global_in': random_input}
    qonnx_output = execute_onnx(model, input_dict, return_full_exec_context=False)
    # print("QONNX output:", qonnx_output[0].shape)
    print("QONNX output:", qonnx_output)

    hls4ml_output = hls_model.predict(random_input)
    
    print("hls4ml output:", hls4ml_output)
    # print("hls4ml output:", hls4ml_output[0].shape)
if __name__ == "__main__":
    main()