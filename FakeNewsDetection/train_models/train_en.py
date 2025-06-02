import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from sklearn.utils import resample
import ast 

# Load CSV Dataset
df = pd.read_csv("/Users/Owner/Documents/codes/fake-news-detection/malay_news_dataset/Processed_Dataset_EN.csv")  

# Preprocess Data
df['label'] = df['Real_Fake'].map({'real': 1, 'fake': 0})
df['text'] = df['text'] = df['Tokenized_Title'].apply(ast.literal_eval).apply(' '.join) + " " + \
             df['Tokenized_Full_Context'].apply(ast.literal_eval).apply(' '.join)
df = df[['text', 'label']] 

#  Oversample fake class to match real class count
fake_df = df[df['label'] == 0]
real_df = df[df['label'] == 1]
if len(fake_df) > 0:
    fake_oversampled = fake_df.sample(len(real_df), replace=True, random_state=42)
    df_balanced = pd.concat([real_df, fake_oversampled]).sample(frac=1, random_state=42)
else:
    df_balanced = df

# Downsample to 5000
df = df_balanced.sample(5000, random_state=42)  # Comment this line to train on all data

# Define EDA Features 
# Top words from EDA
fake_keywords = set(['mahathir', 'umno', 'click', 'covid19', 'patient', 'cluster', 'dr',
                     'noor', 'respiratory', 'live', 'infection', 'nt', 'number', 'lantan',
                     'one', 'get', 'total', 'hisham', 'case'])

real_keywords = set(['court', 'police', 'charge', 'investigation', 'judge', 'government',
                     'lawyer', 'application', 'macc', 'march', 'order', 'appeal', '000',
                     'state', 'section', 'july', 'say', 'file'])

def extract_features(text):
    words = text.lower().split()
    return {
        "has_fake_keyword": int(bool(set(words) & fake_keywords)),
        "has_real_keyword": int(bool(set(words) & real_keywords)),
        "fake_keyword_count": sum(1 for w in words if w in fake_keywords),
        "real_keyword_count": sum(1 for w in words if w in real_keywords),
        "keyword_diff": sum(1 for w in words if w in fake_keywords) - sum(1 for w in words if w in real_keywords),
        "text_length": len(words),
    }

feature_df = pd.DataFrame(df["text"].apply(extract_features).tolist())

# Scale numeric features
scaler = StandardScaler()
scaled = scaler.fit_transform(feature_df[["fake_keyword_count", "real_keyword_count", "keyword_diff", "text_length"]])
feature_df[["fake_keyword_count", "real_keyword_count", "keyword_diff", "text_length"]] = scaled

# Add to main DataFrame
for col in feature_df.columns:
    df[col] = feature_df[col]

# Tokenization + feature mapping 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_with_features(example):
    def safe_float(val, default=0.0):
        return float(val) if val is not None else default

    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = example["label"]
    tokens["additional_features"] = [
        safe_float(example.get("has_fake_keyword")),
        safe_float(example.get("has_real_keyword")),
        safe_float(example.get("fake_keyword_count")),
        safe_float(example.get("real_keyword_count")),
        safe_float(example.get("keyword_diff")),
        safe_float(example.get("text_length")),
        0, 0, 0, 0  # Padding to 10 dims
    ]
    return tokens

# Convert to HuggingFace Dataset
hf_dataset = Dataset.from_pandas(df)
tokenized_dataset = hf_dataset.map(tokenize_with_features)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "additional_features"])

"""
# Load Tokenizer and Model
BERT_MODEL = "distilbert-base-uncased"  # lighter model for faster training
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")
"""
# hybrid model 
class HybridDistilBERTModel(DistilBertPreTrainedModel):
    def __init__(self, config, additional_feature_dim=10):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size + additional_feature_dim, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.relu = nn.ReLU()

    def forward(self, input_ids=None, attention_mask=None, labels=None, additional_features=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = torch.cat((pooled_output, additional_features), dim=1)
        x = self.pre_classifier(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Load config/model
config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
model = HybridDistilBERTModel.from_pretrained("distilbert-base-uncased", config=config)

# collator 
def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.tensor([x['labels'] for x in batch])
    additional_features = torch.stack([x['additional_features'] for x in batch]).float()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'additional_features': additional_features
    }

# Split into Train/Test
split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]

# Define Evaluation Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

""""
# Weighted loss for class imbalance
from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Calculate class weights dynamically
        num_real = (labels == 1).sum().item()
        num_fake = (labels == 0).sum().item()
        if num_fake == 0:
            weights = torch.tensor([1.0, 1.0]).to(logits.device)
        else:
            weights = torch.tensor([float(num_real)/float(num_fake), 1.0]).to(logits.device)
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
"""

# Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Enable if on GPU
    report_to="none",
)


# Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)


# Train
trainer.train()

# Save
model.save_pretrained("/Users/Owner/Documents/codes/fake-news-detection/models/eng_bert_model")
tokenizer.save_pretrained("/Users/Owner/Documents/codes/fake-news-detection/models/eng_bert_model")
print("Model saved to eng_bert-model/")

# Evaluate
eval_results = trainer.evaluate()
print("\nEvaluation Metrics:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# Confusion Matrix 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

predictions = trainer.predict(split["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
print("\nConfusion Matrix:")
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.grid(False)
plt.show() 