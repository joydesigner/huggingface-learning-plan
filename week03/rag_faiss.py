from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# example documents
documents = [
    "The capital of France is Paris.",
    "Deep learning models require large datasets.",
    "FAISS is a library for efficient similarity search."
]

# generate embeddings
doc_embeddings = model.encode(documents, convert_to_tensor=False)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # L2 distance
index.add(doc_embeddings)  # add embeddings to index


# indexing and generating
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def rag_answer(query, top_k=3):
    # generate query embedding
    query_embedding = model.encode(query, convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')

    # search for nearest neighbors
    D, I = index.search(query_embedding.reshape(1, -1), top_k)  # distances and indices

    # retrieve documents
    retrieved_docs = [documents[i] for i in I[0]]
    
    # group the documents
    context = "\n".join(retrieved_docs)
    input_text = f"Answer based on context: {context}\nQuestion: {query}"
    
    # generate answer
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = generator.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    

# Example usage
query = "What is the capital of France?"
answer = rag_answer(query)
print(answer)
