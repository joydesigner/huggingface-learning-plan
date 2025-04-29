from transformers import pipeline
# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering")
# Define the context and question
question = "What is the capital of France?"
context = "The capital of France is Paris."

# Get the answer
result = qa_pipeline(question=question, context=context)
print(f"Question: {question}")
print(f"Answer: {result['answer']}")