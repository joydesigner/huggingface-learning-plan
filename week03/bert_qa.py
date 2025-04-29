from transformers import BertTokenizer, AutoModelForQuestionAnswering
import torch
# This script demonstrates how to use a pre-trained BERT model for question answering.
# It loads a pre-trained BERT model and tokenizer, tokenizes the input question and context,
# and then uses the model to predict the start and end positions of the answer in the context.
# Make sure to install the transformers library if you haven't already
# pip install transformers
# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Define the context and question
context = "The capital of France is Paris."
question = "What is the capital of France?"

# Tokenize the context and question
inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
outputs = model(**inputs)

# Get the start and end scores
start_scores = outputs.start_logits
end_scores = outputs.end_logits
# Get the most likely start and end of the answer
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores) + 1  # +1 because end is inclusive
# Get the answer tokens
answer_tokens = inputs['input_ids'][0][answer_start:answer_end]

# Decode the answer tokens to string
answer = tokenizer.decode(answer_tokens)
print(f"Question: {question}")
print(f"Answer: {answer}")