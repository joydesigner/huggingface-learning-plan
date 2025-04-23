# The huggingface pipeline tasks example
from transformers import pipeline

# sentiment-analysis
classifier = pipeline("sentiment-analysis")
results = classifier("I love using Hugging Face's transformers library!")
print(results)

# Summarization
summarizer = pipeline("summarization")
text = (
    "Hugging Face is creating a tool that democratizes AI by letting you train "
    "and deploy state-of-the-art models powered by the community."
)
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary)

