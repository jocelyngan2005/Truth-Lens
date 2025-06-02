# Truth-Lens 🤖

Truth-Lens is an AI-Powered Telegram bot that helps users identify and verify potentially fake news content. Using advanced machine learning technology, it analyzes news articles, text, and PDF documents to detect misinformation and provide users with confidence scores for each analysis.

## Features 🌟

- **Multi-format Analysis**: Analyze news from URLs, direct text input, or PDF documents
- **Bilingual Support**: Supports both English and Malay languages
- **Real-time Analysis**: Quick and accurate fake news detection
- **User Feedback System**: Collects user feedback to improve accuracy
- **Report System**: Allows users to report misclassified content
- **Confidence Scoring**: Provides confidence levels for each prediction

## Technical Stack 🛠

- **Backend**: Java (Telegram Bot API)
- **ML Service**: Python (Flask)
- **ML Model**: DistilBERT for sequence classification
- **Database**: SQLite for feedback storage
- **Dependencies**:
  - Java: TelegramBots, Gson, SQLite JDBC
  - Python: Flask, Transformers, PyTorch, BeautifulSoup4, PDFPlumber

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
pip install -r src/main/python/requirements.txt
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

1. Start the Python ML service:
```bash
python src/main/python/news_analyzer.py
```

2. Start the Java bot:
```bash
java -jar target/truth-lens-1.0-SNAPSHOT.jar
```

## Usage Guide 📱

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
- All contributors and users of Truth-Lens