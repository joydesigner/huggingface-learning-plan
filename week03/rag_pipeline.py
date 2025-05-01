from datasets import Dataset
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import torch

# Custom knowledge base (can be loaded from CSV, JSON, etc.)
data = {
   "text": [
        "To reset your password, go to the login page and click on 'Forgot Password'.",
        "You can update your billing info under Account > Billing.",
        "To contact support, email help@example.com or call 123-456.",
        "System requirements: at least 8GB RAM and 2GHz CPU."
    ]
}

dataset = Dataset.from_dict(data)

# Use RAG Tokenizer (shared between retriever and generator)
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

# Index knowledge base into vector database
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-base",
    index_name="custom",
    passages=dataset
)
# Load RAG model
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

question = "How do I reset my password?"
input_dict = tokenizer.prepare_seq2seq_batch(
    [question],
    return_tensors="pt"
)

# generate answer using both question and retrieved documents
generated = model.generate(
    input_ids=input_dict["input_ids"],
    attention_mask=input_dict["attention_mask"],
    num_beams=2,
    max_length=50
)

answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(f"Question: {question}")
print(f"Answer: {answer}")