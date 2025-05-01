# Rag Pipeline
## Scenario
Building an AI assistant for a **tech support knowledge base**. When users ask questions (e.g., “How to reset my password?”), the system should:
	1.	Retrieve relevant documents (FAQs, guides) from a local dataset.
	2.	Use a language model (like GPT-2 or Llama) to generate an answer **based on those documents**.

## Stack
- Hugging Face transformers
- Datasets for storing the knowledge base
- faiss for fast document search
- rag-token-base model (for any LLM)
- pytorch

## RAG pipeline Overview  
```graph LR
    Q[User Question] --> R[Retriever]
    R --> D[Relevant Docs]
    D --> G[Generator (LLM)]
    Q --> G
    G --> A[Answer]
```

### Install dependencies
```bash
pip install transformers datasets faiss-cpu
```
