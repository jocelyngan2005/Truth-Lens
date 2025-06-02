from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import json
import os
import logging
import socket
import pdfplumber
import io
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define the same model architectures as in testing scripts
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
        pooled_output = outputs.pooler_output
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

# Define the same keyword sets as in testing scripts
fake_keywords_eng = set(['mahathir', 'umno', 'click', 'covid19', 'patient', 'cluster', 'dr',
                        'noor', 'respiratory', 'live', 'infection', 'nt', 'number', 'lantan',
                        'one', 'get', 'total', 'hisham', 'case'])

real_keywords_eng = set(['court', 'police', 'charge', 'investigation', 'judge', 'government',
                        'lawyer', 'application', 'macc', 'march', 'order', 'appeal', '000',
                        'state', 'section', 'july', 'say', 'file'])

fake_keywords_bm = set(['tular', 'rasmi', 'sosial', 'facebook', 'palsu', 'konon', 'halal', 'dakwa', 
                       'waspada', 'kkm', 'mesej', 'benar', 'sepertimana', 'sebar', 'jakim', 'hospital', 
                       'tipu', 'laman', 'whatsapp', 'nasihat'])

real_keywords_bm = set(['satu', 'ahli', 'anwar', 'kerusi', 'pn', 'calon', 'dia', 'pas', 'quot', 'ph',
                       'pru15', 'pkr', 'gabung', 'bincang', 'undi', 'dap', 'lalu', 'tanding',
                       'muhyiddin', 'sokong'])

def extract_features(text, language='eng'):
    """Extract features using the same method as testing scripts"""
    words = text.lower().split()
    fake_keywords = fake_keywords_eng if language == 'eng' else fake_keywords_bm
    real_keywords = real_keywords_eng if language == 'eng' else real_keywords_bm
    
    return [
        int(bool(set(words) & fake_keywords)),
        int(bool(set(words) & real_keywords)),
        sum(1 for w in words if w in fake_keywords),
        sum(1 for w in words if w in real_keywords),
        sum(1 for w in words if w in fake_keywords) - sum(1 for w in words if w in real_keywords),
        len(words),
        0, 0, 0, 0  # Padding to 10 dims
    ]

# Initialize models dictionary
models = {}

def load_models():
    """Load both English and Malay models"""
    try:
        model_dir = os.path.dirname(__file__)
        
        # Load English model
        logger.info("Loading English BERT model...")
        eng_model_path = os.path.join(model_dir, "models", "eng_bert_model")
        eng_tokenizer = AutoTokenizer.from_pretrained(eng_model_path)
        eng_config = AutoConfig.from_pretrained(eng_model_path)
        eng_model = HybridDistilBERTModel.from_pretrained(
            eng_model_path,
            config=eng_config,
            ignore_mismatched_sizes=True
        )
        eng_model.eval()
        logger.info("Successfully loaded English BERT model")
        
        # Load Malay model
        logger.info("Loading Malay BERT model...")
        bm_model_path = os.path.join(model_dir, "models", "bm_bert_model")
        bm_tokenizer = AutoTokenizer.from_pretrained(bm_model_path)
        bm_config = AutoConfig.from_pretrained(bm_model_path)
        bm_model = HybridBertModel.from_pretrained(
            bm_model_path,
            config=bm_config,
            ignore_mismatched_sizes=True
        )
        bm_model.eval()
        logger.info("Successfully loaded Malay BERT model")
        
        # Store models in dictionary
        models['eng'] = {
            'model': eng_model,
            'tokenizer': eng_tokenizer,
            'config': eng_config
        }
        models['bm'] = {
            'model': bm_model,
            'tokenizer': bm_tokenizer,
            'config': bm_config
        }
        
        logger.info("All models loaded successfully and ready for inference")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def predict_with_bert(text, language='eng'):
    """Make prediction using the appropriate model based on language"""
    try:
        if not models:
            raise Exception("Models not loaded. Please ensure models are loaded before making predictions.")
            
        if language not in ['eng', 'bm']:
            raise ValueError(f"Invalid language: {language}. Must be 'eng' or 'bm'")
            
        model_data = models[language]
        tokenizer = model_data['tokenizer']
        model = model_data['model']
        
        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        # Extract features using the same method as testing scripts
        features = torch.tensor([extract_features(text, language)]).float()
        
        # Get prediction
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                additional_features=features
            )
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get prediction and confidence
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()
        
        return {
            'is_fake': bool(prediction),
            'confidence': float(confidence),
            'probabilities': {
                'fake': float(probabilities[0][0].item()),
                'real': float(probabilities[0][1].item())
            }
        }
    except Exception as e:
        logger.error(f"Error in BERT prediction: {str(e)}")
        raise

@app.route('/analyze', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        url = data.get('url')
        text = data.get('text')
        pdf_content = data.get('pdf_content')
        language = data.get('language', 'eng').lower()  # Default to English if not specified
        
        # Validate language parameter
        if language not in ['eng', 'bm']:
            return jsonify({
                'error': f"Invalid language: {language}. Must be 'eng' or 'bm'"
            }), 400
        
        logger.info(f"Received request - URL: {url}, Text: {text[:100] if text else None}, PDF: {'Yes' if pdf_content else 'No'}, Language: {language}")
        
        # Get content from appropriate source
        if url:
            result = extract_article_content(url)
            if not result['success']:
                return jsonify({'error': result['error']}), 400
            content = result['content']
        elif text:
            content = text
        elif pdf_content:
            try:
                pdf_bytes = base64.b64decode(pdf_content)
                result = extract_text_from_pdf(pdf_bytes)
                if not result['success']:
                    return jsonify({'error': result['error']}), 400
                content = result['content']
            except Exception as e:
                logger.error(f"Error decoding PDF content: {str(e)}")
                return jsonify({'error': f'Invalid PDF content: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No content provided for analysis'}), 400
        
        # Get prediction using the specified language
        try:
            prediction = predict_with_bert(content, language)
            logger.info(f"Successfully analyzed content using {language} model")
        except Exception as e:
            logger.error(f"Error getting BERT prediction: {str(e)}")
            return jsonify({'error': f'Error analyzing content: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'content': content[:500] + '...' if len(content) > 500 else content,
            'language': language,
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_news: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Initialize models when the application starts
if not load_models():
    logger.error("Failed to load models. Application may not function correctly.")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def extract_article_content(url):
    try:
        logger.info(f"Attempting to fetch content from URL: {url}")
        
        # Send request with headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        logger.info(f"Making request to URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        logger.info("Successfully fetched the webpage")
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try different selectors to find the main content
        selectors = [
            'article',
            '.article-content',
            '.story-body',
            '.post-content',
            '.entry-content',
            'main',
            '[role="article"]',
            '.content',
            '#content'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                # Get the text content and clean it
                content = ' '.join(elem.get_text().strip() for elem in elements)
                content = ' '.join(content.split())  # Remove extra whitespace
                
                if len(content) > 100:  # Ensure we have substantial content
                    logger.info(f"Found content using selector: {selector}")
                    return {
                        'success': True,
                        'content': content,
                        'title': soup.title.string if soup.title else None
                    }
        
        # If no specific selectors work, try to get the body text
        logger.info("No specific selectors found, using body text")
        body_text = soup.body.get_text() if soup.body else ''
        body_text = ' '.join(body_text.split())
        
        if len(body_text) < 100:
            logger.warning("Extracted content is too short")
            return {
                'success': False,
                'error': 'Could not extract sufficient content from the webpage'
            }
        
        return {
            'success': True,
            'content': body_text,
            'title': soup.title.string if soup.title else None
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {
            'success': False,
            'error': f'Error accessing the URL: {str(e)}'
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'success': False,
            'error': f'Error processing the content: {str(e)}'
        }

def extract_text_from_pdf(pdf_bytes):
    try:
        logger.info("Attempting to extract text from PDF")
        logger.info(f"PDF size: {len(pdf_bytes)} bytes")
        
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            logger.info(f"PDF opened successfully. Number of pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        logger.info(f"Successfully extracted text from page {i}")
                    else:
                        logger.warning(f"No text found on page {i}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i}: {str(e)}")
        
        if not text.strip():
            logger.warning("No text could be extracted from the PDF")
            return {
                'success': False,
                'error': 'Could not extract text from the PDF. The file might be scanned or contain only images.'
            }
        
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return {
            'success': True,
            'content': text
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {
            'success': False,
            'error': f'Error processing the PDF: {str(e)}'
        }

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # Try ports in order: 5000, 5001, 5002, 5003, 5004
    for port in range(5000, 5005):
        if not is_port_in_use(port):
            logger.info(f"Starting server on port {port}")
            logger.info(f"Server will be available at http://localhost:{port}")
            logger.info(f"Health check endpoint available at http://localhost:{port}/health")
            
            try:
                app.run(host='0.0.0.0', port=port, debug=True)
                break
            except Exception as e:
                logger.error(f"Failed to start server on port {port}: {str(e)}")
                continue
    else:
        logger.error("Could not find an available port between 5000-5004")
        exit(1)