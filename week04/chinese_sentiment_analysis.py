# fine tune Chinese-Bert-wwm-text
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
import torch
import pandas as pd
import os

# 1. 首先检查并解析数据文件格式
# 假设数据格式为"文本 标签"（标签为0或1）

def process_data_files(file_path):
    processed_data = {"text": [], "label": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if i + 2 < len(lines):  # 确保至少有3行可以读取
                    text = lines[i].strip()
                    feature = lines[i+1].strip()  # 特征，如"质量"、"快递"等
                    try:
                        label = int(lines[i+2].strip())
                        # 可以选择是否将特征添加到文本中
                        # processed_data["text"].append(f"{text} 特征：{feature}")
                        processed_data["text"].append(text)
                        processed_data["label"].append(label)
                    except ValueError:
                        # 跳过无效行
                        print(f"警告: 无效的标签 '{lines[i+2].strip()}' 在行 {i+3}")
                    i += 3
                else:
                    # 如果剩余行数不足3行，跳出循环
                    break
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return pd.DataFrame(processed_data)

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建数据文件的绝对路径
train_path = os.path.join(current_dir, "data", "train.txt")
test_path = os.path.join(current_dir, "data", "test.txt")

# 检查文件是否存在
if not os.path.exists(train_path):
    raise FileNotFoundError(f"训练文件不存在: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"测试文件不存在: {test_path}")

print(f"加载训练数据: {train_path}")
print(f"加载测试数据: {test_path}")

# 处理训练和测试数据
train_data = process_data_files(train_path)
test_data = process_data_files(test_path)

# 检查数据是否为空
if len(train_data) == 0:
    raise ValueError(f"训练数据集为空，请检查文件: {train_path}")
if len(test_data) == 0:
    raise ValueError(f"测试数据集为空，请检查文件: {test_path}")

print(f"加载训练样本: {len(train_data)} 条")
print(f"加载测试样本: {len(test_data)} 条")

# 2. 将处理后的数据转换为datasets格式
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
dataset = {"train": train_dataset, "test": test_dataset}

# 3. 使用tokenizer预处理文本
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

def tokenize_fn(examples):
    return tokenizer(examples["text"], 
                     padding="max_length", 
                     truncation=True, 
                     max_length=128)

# 4. 应用预处理
tokenized_datasets = {
    split: dataset[split].map(
        tokenize_fn, 
        batched=True, 
        remove_columns=["text"]
    ) 
    for split in dataset
}

# 5. 使用适当的数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. 加载模型
model = BertForSequenceClassification.from_pretrained("hfl/chinese-bert-wwm-ext", num_labels=2)

# 7. 配置训练参数
# 适用于旧版 transformers 的配置
training_args = TrainingArguments(
    output_dir=os.path.join(current_dir, "results"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(current_dir, "logs"),
    logging_steps=10,
    do_eval=True,  # 执行评估
)

# 8. 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# 9. 开始训练
print("开始训练模型...")
trainer.train()

# 10. 评估模型
results = trainer.evaluate()
print(f"评估结果: {results}")

# 11. 保存模型
model_path = os.path.join(current_dir, "chinese-sentiment-model")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"模型已保存至 {model_path}")