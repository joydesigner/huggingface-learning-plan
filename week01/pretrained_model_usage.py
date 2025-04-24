# # download model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# # Example 1: Load BERT 
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Example 2:  Load GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example 3: using Model to inference
## text classification (BERT)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# # output the feature vector
# print(outputs.last_hidden_state.shape) # (1, 7, 768)

# Example 4: Text generation (GPT-2)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

