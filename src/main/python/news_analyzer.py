from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import json
import os
import logging
import socket
import pdfplumber
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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

@app.route('/analyze', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        url = data.get('url')
        text = data.get('text')
        pdf_content = data.get('pdf_content')  # Base64 encoded PDF content
        
        logger.info(f"Received request - URL: {url}, Text: {text[:100] if text else None}, PDF: {'Yes' if pdf_content else 'No'}")
        
        if url:
            # Extract content from URL
            result = extract_article_content(url)
            if not result['success']:
                logger.error(f"Failed to extract content: {result.get('error')}")
                return jsonify({'error': result['error']}), 400
            content = result['content']
        elif text:
            content = text
        elif pdf_content:
            # Decode base64 PDF content
            import base64
            try:
                logger.info("Decoding base64 PDF content")
                pdf_bytes = base64.b64decode(pdf_content)
                logger.info(f"Successfully decoded PDF content, size: {len(pdf_bytes)} bytes")
                result = extract_text_from_pdf(pdf_bytes)
                if not result['success']:
                    logger.error(f"Failed to extract PDF content: {result.get('error')}")
                    return jsonify({'error': result['error']}), 400
                content = result['content']
            except Exception as e:
                logger.error(f"Error decoding PDF content: {str(e)}")
                return jsonify({'error': f'Invalid PDF content: {str(e)}'}), 400
        else:
            logger.error("No URL, text, or PDF content provided")
            return jsonify({'error': 'No content provided for analysis'}), 400
        
        # TODO: Replace this with your actual BERT model prediction
        # This is a temporary random prediction
        import random
        prediction = {
            'is_fake': random.choice([True, False]),
            'confidence': random.uniform(0.5, 1.0)
        }
        
        logger.info("Successfully processed the request")
        return jsonify({
            'success': True,
            'content': content[:500] + '...' if len(content) > 500 else content,
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_news: {str(e)}")
        return jsonify({'error': str(e)}), 500

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