from transformers import AutoTokenizer
import torch
import numpy as np

from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

class HybridBertModel(BertPreTrainedModel):
    def __init__(self, config, additional_feature_dim=10):
        super().__init__(config)
        self.bert = BertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size + additional_feature_dim, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, additional_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch, hidden]
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

    
fake_keywords = set(['tular', 'rasmi', 'sosial', 'facebook', 'palsu', 'konon', 'halal', 'dakwa', 
                     'waspada', 'kkm', 'mesej', 'benar', 'sepertimana', 'sebar', 'jakim', 'hospital', 
                     'tipu', 'laman', 'whatsapp', 'nasihat']
)

real_keywords = set(['satu', 'ahli', 'anwar', 'kerusi', 'pn', 'calon', 'dia', 'pas', 'quot', 'ph',
                     'pru15', 'pkr', 'gabung', 'bincang', 'undi', 'dap', 'lalu', 'tanding',
                     'muhyiddin', 'sokong'])


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
model_dir = "/Users/Owner/Documents/codes/fake-news-detection/models/bm_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = HybridBertModel.from_pretrained(model_dir)

# Example text to predict
texts = [
    "Bantuan Tunai RM500 Untuk Semua Rakyat Mulai Esok – Tanpa Perlu Pengesahan!", 
    "Solat Jumaat Wajib Bawa Kad Pengenalan, Jika Tidak RM500 Denda",  
    "Lazada & Shopee Akan Kenakan Caj 1% Untuk Semua Produk Bawah RM100 Mulai Julai", 
    "GST Akan Dikenakan 1% Untuk Pembelian Dalam Talian Bawah RM50"

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
