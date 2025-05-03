from transformers import BertTokenizer, BertForSequenceClassification
import torch.onnx
import onnxruntime as ort

model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

dummy_input = tokenizer("测试文本", return_tensors="pt")
torch.onnx.export(
    model,
    tuple(dummy_input.values()),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"}
    }
)

# ONNX Runtime inference
sess = ort.InferenceSession("model.onnx")
outputs = sess.run(None, {
    "input_ids": dummy_input["input_ids"].numpy(),
    "attention_mask": dummy_input["attention_mask"].numpy()
})