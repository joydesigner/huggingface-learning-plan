from transformers import pipeline, AutoConfig

model_path = "results/checkpoint-6250"
config = AutoConfig.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

result = classifier("This movie was bad!")
print(result)  # Output: [{'label': 'POSITIVE', 'score': 0.99}]

label_map = {0: "NEGATIVE", 1: "POSITIVE"}
predicted_class = label_map[int(result[0]['label'].split("_")[-1])] 
print(predicted_class)  # Output: POSITIVE