# Truth-Lens 🤖

Truth-Lens is an AI-Powered platform that helps users identify and verify potentially fake news content. It's available both as a Telegram bot and a web application. Using advanced machine learning technology, it analyzes news articles, text, and PDF documents to detect misinformation and provide users with confidence scores for each analysis.

## Features 🌟

- **Multi-format Analysis**: Analyze news from URLs, direct text input, or PDF documents
- **Bilingual Support**: Supports both English and Malay languages
- **Real-time Analysis**: Quick and accurate fake news detection
- **User Feedback System**: Collects user feedback to improve accuracy
- **Report System**: Allows users to report misclassified content
- **Confidence Scoring**: Provides confidence levels for each prediction
- **Multiple Platforms**: Available as both Telegram bot and web application

## Technical Stack 🛠

- **Backend**: 
  - Java (Telegram Bot API)
  - Python (Flask for ML service)
  - Streamlit (Web Application)
- **ML Model**: DistilBERT for sequence classification
- **Database**: SQLite for feedback storage
- **Dependencies**:
  - Java: TelegramBots, Gson, SQLite JDBC
  - Python: Flask, Transformers, PyTorch, BeautifulSoup4, PDFPlumber, Streamlit

## Setup Instructions 📋

### Prerequisites

- Java 11 or higher
- Python 3.8 or higher
- Telegram Bot Token (from BotFather)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jocelyngan2005/Truth-Lens.git
cd Truth-Lens
```

2. Install Python dependencies:
```bash
pip install -r TELEGRAMBOT/src/main/python/requirements.txt
pip install -r STREAMLIT/requirements.txt
```

3. Build the Java project:
```bash
mvn clean install
```

4. Configure your bot:
   - Create a `.env` file in the root directory
   - Add your Telegram bot token:
   ```
   BOT_TOKEN=your_bot_token_here
   BOT_USERNAME=your_bot_username
   ```

### Running the Application

#### Telegram Bot
1. Start the Python ML service:
```bash
python TELEGRAMBOT/src/main/python/news_analyzer.py
```

2. Start the Java bot:
```bash
java -jar target/truth-lens-1.0-SNAPSHOT.jar
```

#### Web Application (Streamlit)
1. Navigate to the Streamlit directory:
```bash
cd STREAMLIT
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the web application at `http://localhost:8501`

### Deployment

#### Telegram Bot
- Deploy the Java bot to your preferred hosting service
- Ensure the Python ML service is running and accessible

#### Web Application
- Deploy to Streamlit Cloud:
  1. Create an account at [share.streamlit.io](https://share.streamlit.io)
  2. Connect your GitHub repository
  3. Deploy from the STREAMLIT directory
  4. Get your HTTPS URL (e.g., `https://yourusername-truth-lens-app-xxxx.streamlit.app`)

## Usage Guide 📱

### Telegram Bot
1. Start a chat with your bot on Telegram
2. Send `/start` to begin
3. Choose your preferred language (English or Malay)
4. Send content for analysis:
   - Forward a message
   - Send a URL
   - Type or paste text
   - Upload a PDF document
5. Receive analysis results with:
   - Prediction (Fake/Real)
   - Confidence score
   - Feedback options

### Web Application
1. Open the Streamlit web application
2. Choose your input method:
   - Text Input: Paste news article directly
   - URL Input: Enter the news article URL
   - PDF Upload: Upload a PDF document
3. Click "Analyze" to get results
4. View detailed analysis including:
   - Fake/Real probability
   - Confidence scores
   - Analysis summary
   - Recommendations
5. Provide feedback on the prediction

## Feedback System 💬

Users can provide feedback on predictions through:
- 👍 Accurate: When the prediction is correct
- 👎 Inaccurate: When the prediction is wrong
- 🚨 Report: To report misclassified content

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Telegram Bot API
- Streamlit
- All future contributors and users of Truth-Lens
- Contributors: 
   - Jocelyn Gan Xin Yi
   - Tan Yin June
   - Cheah Pui Yan