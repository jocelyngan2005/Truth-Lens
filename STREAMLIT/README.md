# TruthLens Web Application 🔍

TruthLens is a web application that helps users identify and verify potentially fake news content. Using advanced machine learning technology, it analyzes news articles, text, and PDF documents to detect misinformation and provide users with confidence scores for each analysis.

## Features 🌟

- **Multi-format Analysis**: Analyze news from URLs, direct text input, or PDF documents
- **Real-time Analysis**: Quick and accurate fake news detection
- **User Feedback System**: Collects user feedback to improve accuracy
- **Report System**: Allows users to report misclassified content to official Malaysian fact-checking platforms
- **Confidence Scoring**: Provides confidence levels for each prediction

## Technical Stack 🛠

- **Frontend**: Streamlit
- **ML Model**: DistilBERT for sequence classification
- **Dependencies**:
  - Python: Streamlit, Transformers, PyTorch, BeautifulSoup4, pdfplumber
  - Database: SQLite for feedback storage

## Setup Instructions 📋

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Navigate to the STREAMLIT directory:
```bash
cd STREAMLIT
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. The application will open in your default web browser at `http://localhost:8501`

## Usage Guide 📱

1. Choose your preferred input method:
   - **Text Input**: Paste your news article directly
   - **URL Input**: Enter the URL of the news article
   - **PDF Upload**: Upload a PDF document containing the news

2. Click the "Analyze" button for your chosen input method

3. View the analysis results:
   - Prediction (Fake/Real)
   - Confidence score
   - Feedback options

## Feedback System 💬

Users can provide feedback on predictions through:
- 👍 Like: When the prediction is accurate
- 👎 Dislike: When the prediction is inaccurate
- ⚠️ Report: To report misclassified content

## Report System 🚨

The application provides direct links to official Malaysian fact-checking platforms:
- MyCheck
- Sebenarnya.my
- PDRM Cyber Crime
- MCMC

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Streamlit
- All contributors and users of TruthLens 