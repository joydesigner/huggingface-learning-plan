from transformers import T5Tokenizer, T5ForConditionalGeneration

# load pre-trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# set padding token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

# encode input text
input_text = "translate English to French: The house is wonderful."
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

# generate translation
outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=50, num_beams=4)

# decode generated text
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
