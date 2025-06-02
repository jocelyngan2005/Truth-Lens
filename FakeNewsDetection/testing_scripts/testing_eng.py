from transformers import AutoTokenizer
import torch
import numpy as np

from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

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
    
fake_keywords = set(['mahathir', 'umno', 'click', 'covid19', 'patient', 'cluster', 'dr',
                     'noor', 'respiratory', 'live', 'infection', 'nt', 'number', 'lantan',
                     'one', 'get', 'total', 'hisham', 'case'])
real_keywords = set(['court', 'police', 'charge', 'investigation', 'judge', 'government',
                     'lawyer', 'application', 'macc', 'march', 'order', 'appeal', '000',
                     'state', 'section', 'july', 'say', 'file'])

def extract_features(text):
    words = text.lower().split()
    return [
        int(bool(set(words) & fake_keywords)),
        int(bool(set(words) & real_keywords)),
        sum(1 for w in words if w in fake_keywords),
        sum(1 for w in words if w in real_keywords),
        sum(1 for w in words if w in fake_keywords) - sum(1 for w in words if w in real_keywords),
        len(words),
        0, 0, 0, 0  # Padding to 10 dims
    ]


# Load the saved model and tokenizer
model_dir = "/Users/Owner/Documents/codes/fake-news-detection/models/eng_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = HybridDistilBERTModel.from_pretrained(model_dir)

# Example text to predict
texts = [
    "Grab Drivers to Charge Double Fares During Rainy Season, Says Transport Minister!", 
    "Health Ministry Confirms New COVID-19 Variant in Malaysia Resistant to All Vaccines", 
    "Malaysian Police Confirm Rise in Kidnappings via WhatsApp Video Calls!",
    "Police Will Arrest Anyone Sharing ‘Negative’ WhatsApp Messages About Economy"

]

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
# Extract features
features = torch.tensor([extract_features(text) for text in texts]).float()

# Predict
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        additional_features=features
    )
    logits = outputs.logits
    preds = np.argmax(logits.numpy(), axis=1)

# Map predictions to labels
label_map = {1: "real", 0: "fake"}
probs = torch.softmax(torch.from_numpy(logits.numpy()), dim=1)
for text, pred, prob in zip(texts, preds, probs):
    confidence = prob[pred].item()
    print(f"Text: {text}\nPrediction: {label_map[pred]}\nConfidence: {confidence:.4f}\n")
