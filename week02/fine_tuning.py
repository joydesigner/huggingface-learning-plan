# load IMDB movie reviews dataset 
# Dataset auto divides into train and test sets
# Each example is a dictionary with 'text' and 'label' keys
# Support Streaming to load big datasets, such as (Stream=True)

# Notice: padding="max_length" is to pad all sequences to the same length, here is the max length of the model (BERT by default is 512)
# batched=True is to process multiple examples at once
from datasets import load_dataset
imdb = load_dataset("imdb")
# print(imdb)

# Data preprocessing
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
  
# Tokenize the dataset
tokenized_imdb = imdb.map(preprocess_function, batched=True)

# Prepare training 
# DataCollator: DataCollator is used to collate the data into batches
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load pretrained model
from transformers import AutoModelForSequenceClassification
# IMDB is a binary classification task, so we use AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# set up training arguments
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    num_train_epochs=2,              # total number of training epochs
    save_steps=500,                  # save checkpoint every 500 steps
    save_total_limit=2,              # limit the total amount of checkpoints
    # 使用 eval_steps 替代 evaluation_strategy
    eval_steps=500,                  # evaluate every 500 steps
    # ensures each eval_steps step is evaluated
    do_eval=True,                    # execute evaluation during training
    logging_dir="./logs",            # directory for storing logs
)

# initialize Trainer
from transformers import Trainer
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_imdb["train"],         # training dataset
    eval_dataset=tokenized_imdb["test"],           # evaluation dataset
    tokenizer=tokenizer,                  # tokenizer
    data_collator=data_collator,         # data collator
)

# fine-tuning and evaluation
trainer.train() # start training
trainer.evaluate() # evaluate the model on the test set