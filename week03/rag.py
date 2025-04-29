
# 加载RAG模型
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# 生成答案
inputs = tokenizer("What is capital of France?", return_tensors="pt")
generated_ids = model.generate(input_ids=inputs["input_ids"])
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
