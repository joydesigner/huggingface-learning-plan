# Sentiment Analysis on Chinese Text
# Web API/ Streamlit
from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()
model_path = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model= AutoModelForSequenceClassification.from_pretrained(model_path)
# add padding token
tokenizer.pad_token = tokenizer.eos_token

@app.post("/predict")
async def predict(text: str):
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
  with torch.no_grad():
    outputs = model(**inputs)
  probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
  return {"positive": probs[0][1].item(), "negative": probs[0][0].item()}

# To run the FastAPI app, use the command:
# uvicorn chinese_sentiment_analysis:app --reload

