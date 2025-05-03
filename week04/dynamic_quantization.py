from transformers import AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx",
    "model_quant.onnx",
    weight_type=QuantType.QInt8
)