# huggingface-learning-plan
A systematic personal huggingface learning and play around plan help anyone to get started easily and quickly.

# 🤗 Hugging Face 学习与实战计划

一个为期 4 周的系统学习计划，涵盖 Hugging Face Transformers、Datasets、模型微调、问答系统、部署与优化。

---

## 📌 项目结构建议（GitHub Repo）
```
📁 huggingface-learning-plan
├── README.md              # 项目说明与学习路线
├── week1_intro/
│   ├── sentiment_pipeline.py
│   └── summarization_pipeline.py
├── week2_finetuning/
│   ├── imdb_finetune.py
│   └── dataset_demo.py
├── week3_advanced_tasks/
│   ├── question_answering.py
│   ├── text_generation.py
│   └── rag_demo/
│       ├── faiss_retriever.py
│       └── rag_pipeline.py
├── week4_deploy_optimize/
│   ├── fastapi_deploy.py
│   ├── gradio_demo.py
│   └── model_quantization.py
├── requirements.txt       # 所需依赖
└── .gitignore
```

---

## 🗂️ 任务清单与学习卡片（Notion）
> 你可以将以下内容复制粘贴进你的 Notion 页面，按勾选方式管理学习进度：

### ✅ 学习计划（Notion模板）

#### 📅 Week 1：快速入门与模型调用
- [ ] 安装 Transformers, Datasets, Accelerate
- [ ] 使用 `pipeline` 跑通情感分析、摘要、翻译任务
- [ ] 浏览 Hugging Face Hub，下载模型
- [ ] 阅读官方 QuickTour 教程

#### 📅 Week 2：微调与数据集处理
- [ ] 熟悉 `datasets` 加载 IMDB/AG News
- [ ] 使用 `Trainer` + BERT 完成文本分类微调
- [ ] 上传微调模型至 Hugging Face Hub

#### 📅 Week 3：复杂任务实现
- [ ] 使用 T5/GPT2 进行文本生成
- [ ] 使用 BERT 问答模型完成问答任务
- [ ] 搭建一个 RAG + FAISS 问答系统

#### 📅 Week 4：部署与优化
- [ ] 使用 FastAPI 将模型封装成 REST API
- [ ] 使用 Gradio 打造交互式界面
- [ ] 使用 Optimum + ONNX 进行模型加速
- [ ] 尝试部署到 Hugging Face Spaces（可选）

---

## 📘 推荐命名规范
- `README.md` 用于说明整个项目结构和使用指南
- 每周的 `.py` 文件按任务命名
- 加入 `requirements.txt` 管理依赖

---
