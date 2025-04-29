from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# explicitly set padding token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)
# Generate text
output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=100, num_beams=5, no_repeat_ngram_size=2. temp=0.7, early_stopping=True, repeat_penalty=1.2)
# Decode generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)