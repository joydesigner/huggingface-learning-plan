import streamlit as st
from transformers import pipeline

st.title("中文情感分析")
model = pipeline("text-classification", model="bert-base-chinese")

text = st.text_area("输入淘宝评论:")
if st.button("分析"):
    result = model(text)[0]
    st.write(f"标签: {result['label']}, 置信度: {result['score']:.2f}")
    # read the label from the model
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    predicted_class = label_map[int(result['label'].split("_")[-1])]
    st.write(f"预测类别: {predicted_class}")
    # read the label from the model