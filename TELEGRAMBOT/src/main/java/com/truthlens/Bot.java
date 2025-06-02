package com.truthlens;

import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.objects.Update;
import org.telegram.telegrambots.meta.api.objects.Message;
import org.telegram.telegrambots.meta.api.objects.replykeyboard.InlineKeyboardMarkup;
import org.telegram.telegrambots.meta.api.objects.replykeyboard.buttons.InlineKeyboardButton;
import org.telegram.telegrambots.meta.exceptions.TelegramApiException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.time.Duration;
import org.telegram.telegrambots.meta.api.methods.GetFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.sql.*;
import java.util.Timer;
import java.util.TimerTask;

public class Bot extends TelegramLongPollingBot {
    private final String botToken;
    private final String botUsername;
    private final HttpClient httpClient;
    private final Gson gson;
    private static final String[] PYTHON_SERVICE_URLS = {
        "http://localhost:5000/analyze",
        "http://localhost:5001/analyze",
        "http://localhost:5002/analyze",
        "http://localhost:5003/analyze",
        "http://localhost:5004/analyze"
    };
    private final Map<Long, PendingReport> pendingReports = new ConcurrentHashMap<>();
    private final Map<Long, String> pendingLanguage = new ConcurrentHashMap<>();
    private final Connection dbConnection;
    private Map<Long, Timer> feedbackTimers = new ConcurrentHashMap<>();

    private static class PendingReport {
        final String contentHash;
        final boolean isFake;

        PendingReport(String contentHash, boolean isFake) {
            this.contentHash = contentHash;
            this.isFake = isFake;
        }
    }

    @SuppressWarnings("deprecation")
    public Bot(String botToken, String botUsername) {
        this.botToken = botToken;
        this.botUsername = botUsername;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
        this.gson = new Gson();
        
        // Initialize database
        try {
            Class.forName("org.sqlite.JDBC");
            this.dbConnection = DriverManager.getConnection("jdbc:sqlite:truthlens.db");
            initializeDatabase();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize database", e);
        }
    }

    private void initializeDatabase() throws SQLException {
        try (Statement stmt = dbConnection.createStatement()) {
            // Create feedback table
            stmt.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    action TEXT NOT NULL,
                    is_fake BOOLEAN NOT NULL,
                    language TEXT,
                    reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """);
        }
    }

    private void storeFeedback(String contentHash, String action, boolean isFake, String language, String reason) {
        try (PreparedStatement pstmt = dbConnection.prepareStatement(
                "INSERT INTO feedback (content_hash, action, is_fake, language, reason) VALUES (?, ?, ?, ?, ?)")) {
            pstmt.setString(1, contentHash);
            pstmt.setString(2, action);
            pstmt.setBoolean(3, isFake);
            pstmt.setString(4, language);
            pstmt.setString(5, reason);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            System.err.println("Error storing feedback: " + e.getMessage());
        }
    }

    @Override
    public String getBotUsername() {
        return botUsername;
    }
  
    @Override
    public String getBotToken() {
        return botToken;
    }
  
    @Override
    public void onUpdateReceived(Update update) {
        if (update.hasCallbackQuery()) {
            handleCallbackQuery(update.getCallbackQuery());
            return;
        }

        if (update.hasMessage()) {
            Message message = update.getMessage();
            long chatId = message.getChatId();

            // Check if this is a report reason
            if (pendingReports.containsKey(chatId)) {
                handleReportReason(message);
                return;
            }

            // Handle /start command
            if (message.hasText() && message.getText().equals("/start")) {
                handleStartCommand(chatId);
                return;
            }

            // Check if we're waiting for language selection
            if (pendingLanguage.containsKey(chatId)) {
                handleLanguageSelection(message);
                return;
            }

            // Handle forwarded messages
            if (message.getForwardFrom() != null || message.getForwardFromChat() != null) {
                showLanguageSelection(message, chatId);
            }
            // Handle direct text messages
            else if (message.hasText()) {
                showLanguageSelection(message, chatId);
            }
            // Handle documents (PDFs)
            else if (message.hasDocument()) {
                showLanguageSelection(message, chatId);
            }
        }
    }

    private void handleStartCommand(long chatId) {
        String welcomeMessage = 
            "👋 *Welcome to TruthLens – Your Fake News Detective!*\n\n" +
            "I'm here to help you spot misinformation and verify news in seconds. Here's how I can assist:\n\n" +
            "🔍 Send me a news headline, article, or link – I'll analyze it for credibility.\n" +
            "📢 Forward suspicious messages – I'll fact-check claims in real time.\n" +
            "📌 Paste text snippets – I'll detect red flags like bias or false sources.\n\n" +
            "Let's fight misinformation together! Send me anything you'd like verified. ✨";

        sendMessage(chatId, welcomeMessage);

        // Show language selection immediately
        InlineKeyboardMarkup markup = new InlineKeyboardMarkup();
        List<List<InlineKeyboardButton>> keyboard = new ArrayList<>();
        List<InlineKeyboardButton> row = new ArrayList<>();

        // English button
        InlineKeyboardButton englishButton = new InlineKeyboardButton();
        englishButton.setText("🇬🇧 English");
        englishButton.setCallbackData("lang:en:0");
        row.add(englishButton);

        // Malay button
        InlineKeyboardButton malayButton = new InlineKeyboardButton();
        malayButton.setText("🇲🇾 Malay");
        malayButton.setCallbackData("lang:ms:0");
        row.add(malayButton);

        keyboard.add(row);
        markup.setKeyboard(keyboard);

        SendMessage languageMessage = new SendMessage();
        languageMessage.setChatId(chatId);
        languageMessage.setText("🌐 Please select your preferred language:");
        languageMessage.setReplyMarkup(markup);

        try {
            execute(languageMessage);
        } catch (TelegramApiException e) {
            e.printStackTrace();
        }
    }

    private void showLanguageSelection(Message message, long chatId) {
        InlineKeyboardMarkup markup = new InlineKeyboardMarkup();
        List<List<InlineKeyboardButton>> keyboard = new ArrayList<>();
        List<InlineKeyboardButton> row = new ArrayList<>();

        // English button
        InlineKeyboardButton englishButton = new InlineKeyboardButton();
        englishButton.setText("🇬🇧 English");
        englishButton.setCallbackData("lang:en:" + message.getMessageId());
        row.add(englishButton);

        // Malay button
        InlineKeyboardButton malayButton = new InlineKeyboardButton();
        malayButton.setText("🇲🇾 Malay");
        malayButton.setCallbackData("lang:ms:" + message.getMessageId());
        row.add(malayButton);

        keyboard.add(row);
        markup.setKeyboard(keyboard);

        SendMessage languageMessage = new SendMessage();
        languageMessage.setChatId(chatId);
        languageMessage.setText("🌐 Please select the language of the content:");
        languageMessage.setReplyMarkup(markup);

        try {
            execute(languageMessage);
        } catch (TelegramApiException e) {
            e.printStackTrace();
        }
    }

    private void handleLanguageSelection(Message message) {
        long chatId = message.getChatId();
        String language = pendingLanguage.get(chatId);
        
        if (language != null) {
            // Process the content based on the selected language
            if (message.hasText()) {
                if (message.getText().startsWith("http")) {
                    handleUrlMessage(message.getText(), chatId, language);
                } else {
                    analyzeContent(message.getText(), chatId, language);
                }
            } else if (message.hasDocument()) {
                handleDocument(message, chatId, language);
            }
            
            // Clear the pending language
            pendingLanguage.remove(chatId);
        }
    }

    private void handleUrlMessage(String url, long chatId, String language) {
        Exception lastException = null;
        
        // Try each URL until one works
        for (String serviceUrl : PYTHON_SERVICE_URLS) {
            try {
                sendMessage(chatId, "🔍 Analyzing the article... Please wait.");
                
                // Create request body
                JsonObject requestBody = new JsonObject();
                requestBody.addProperty("url", url);
                
                // Send request to Python service
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serviceUrl))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                    .timeout(Duration.ofSeconds(30))
                    .build();
                
                HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
                
                if (response.statusCode() == 200) {
                    JsonObject result = gson.fromJson(response.body(), JsonObject.class);
                    if (result.get("success").getAsBoolean()) {
                        String content = result.get("content").getAsString();
                        JsonObject prediction = result.getAsJsonObject("prediction");
                        boolean isFake = prediction.get("is_fake").getAsBoolean();
                        double confidence = prediction.get("confidence").getAsDouble();
                        
                        sendAnalysisResult(chatId, content, isFake, confidence);
                        return; // Success, exit the method
                    } else {
                        String error = result.has("error") ? result.get("error").getAsString() : "Unknown error";
                        sendMessage(chatId, "❌ Error analyzing the article: " + error);
                        return; // Error from service, exit the method
                    }
                }
            } catch (Exception e) {
                lastException = e;
                // Continue to next URL
            }
        }
        
        // If we get here, all URLs failed
        if (lastException instanceof java.net.ConnectException) {
            sendMessage(chatId, "❌ Could not connect to the analysis service. Please make sure the Python service is running.");
        } else if (lastException instanceof java.net.http.HttpTimeoutException) {
            sendMessage(chatId, "❌ The request timed out. Please try again later.");
        } else {
            sendMessage(chatId, "❌ Error processing the URL. Please make sure it's a valid news article URL.");
            lastException.printStackTrace();
        }
    }

    private void analyzeContent(String text, long chatId, String language) {
        try {
            // Send loading message
            sendMessage(chatId, "🔍 Analyzing the content... Please wait.");
            
            // Create request body
            JsonObject requestBody = new JsonObject();
            requestBody.addProperty("text", text);
            
            // Send request to Python service
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(PYTHON_SERVICE_URLS[0]))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .timeout(Duration.ofSeconds(30))
                .build();
            
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() == 200) {
                JsonObject result = gson.fromJson(response.body(), JsonObject.class);
                if (result.get("success").getAsBoolean()) {
                    JsonObject prediction = result.getAsJsonObject("prediction");
                    boolean isFake = prediction.get("is_fake").getAsBoolean();
                    double confidence = prediction.get("confidence").getAsDouble();
                    
                    sendAnalysisResult(chatId, text, isFake, confidence);
                } else {
                    String error = result.has("error") ? result.get("error").getAsString() : "Unknown error";
                    sendMessage(chatId, "❌ Error analyzing the content: " + error);
                }
            } else {
                sendMessage(chatId, "❌ Error connecting to the analysis service. Please try again later.");
            }
        } catch (java.net.ConnectException e) {
            sendMessage(chatId, "❌ Could not connect to the analysis service. Please make sure the Python service is running.");
        } catch (java.net.http.HttpTimeoutException e) {
            sendMessage(chatId, "❌ The request timed out. Please try again later.");
        } catch (Exception e) {
            sendMessage(chatId, "Sorry, I encountered an error while analyzing the content. Please try again.");
            e.printStackTrace();
        }
    }

    private void handleDocument(Message message, long chatId, String language) {
        try {
            // Check if it's a PDF
            if (!message.getDocument().getFileName().toLowerCase().endsWith(".pdf")) {
                sendMessage(chatId, "❌ Please send a PDF file.");
                return;
            }

            sendMessage(chatId, "📄 Processing your PDF... Please wait.");

            // Get the file
            String fileId = message.getDocument().getFileId();
            GetFile getFile = new GetFile();
            getFile.setFileId(fileId);
            org.telegram.telegrambots.meta.api.objects.File file = execute(getFile);
            
            // Download the file
            String fileUrl = "https://api.telegram.org/file/bot" + botToken + "/" + file.getFilePath();
            @SuppressWarnings("deprecation")
            java.net.URL url = new java.net.URL(fileUrl);
            java.io.InputStream in = url.openStream();
            byte[] pdfBytes = in.readAllBytes();
            in.close();
            
            if (pdfBytes.length == 0) {
                sendMessage(chatId, "❌ Error: The PDF file is empty. Please send a valid PDF file.");
                return;
            }
            
            // Convert to base64
            String base64Pdf = java.util.Base64.getEncoder().encodeToString(pdfBytes);
            
            // Create request body
            JsonObject requestBody = new JsonObject();
            requestBody.addProperty("pdf_content", base64Pdf);
            
            // Send request to Python service
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(PYTHON_SERVICE_URLS[0]))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .timeout(Duration.ofSeconds(30))
                .build();
            
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() == 200) {
                JsonObject result = gson.fromJson(response.body(), JsonObject.class);
                if (result.get("success").getAsBoolean()) {
                    String content = result.get("content").getAsString();
                    JsonObject prediction = result.getAsJsonObject("prediction");
                    boolean isFake = prediction.get("is_fake").getAsBoolean();
                    double confidence = prediction.get("confidence").getAsDouble();
                    
                    sendAnalysisResult(chatId, content, isFake, confidence);
                } else {
                    String error = result.has("error") ? result.get("error").getAsString() : "Unknown error";
                    sendMessage(chatId, "❌ Error analyzing the PDF: " + error);
                }
            } else {
                sendMessage(chatId, "❌ Error connecting to the analysis service. Please try again later.");
            }
        } catch (java.net.ConnectException e) {
            sendMessage(chatId, "❌ Could not connect to the analysis service. Please make sure the Python service is running.");
        } catch (java.net.http.HttpTimeoutException e) {
            sendMessage(chatId, "❌ The request timed out. Please try again later.");
        } catch (Exception e) {
            sendMessage(chatId, "❌ Error processing the PDF: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void handleReportReason(Message message) {
        long chatId = message.getChatId();
        PendingReport report = pendingReports.get(chatId);
        
        if (report != null) {
            String reason = message.getText();
            
            // Store the report in database
            storeFeedback(report.contentHash, "report", report.isFake, null, reason);
            
            // Send confirmation
            String confirmationMessage = 
                "🚨 *Report Submitted*\n\n" +
                "Thank you for your report. Our team will review this content.\n\n" +
                "Your reason: " + reason;
            
            sendMessage(chatId, confirmationMessage);
            
            // Remove the pending report
            pendingReports.remove(chatId);

            // Cancel any existing timer
            Timer existingTimer = feedbackTimers.remove(chatId);
            if (existingTimer != null) {
                existingTimer.cancel();
            }

            // Show language selection buttons after report submission
            showLanguageSelectionButtons(chatId);
        }
    }

    private void handleCallbackQuery(org.telegram.telegrambots.meta.api.objects.CallbackQuery callbackQuery) {
        String callbackData = callbackQuery.getData();
        long chatId = callbackQuery.getMessage().getChatId();

        if (callbackData.startsWith("lang:")) {
            // Handle language selection
            String[] parts = callbackData.split(":");
            String language = parts[1];
            
            // Store the selected language
            pendingLanguage.put(chatId, language);
            
            // Send confirmation
            String languageName = language.equals("en") ? "English" : "Malay";
            sendMessage(chatId, "✅ Selected language: " + languageName + "\nPlease send your content for analysis.");
            return;
        }

        // Parse the callback data
        String[] parts = callbackData.split(":");
        if (parts.length != 3) return;

        String action = parts[0];
        String contentHash = parts[1];
        boolean isFake = Boolean.parseBoolean(parts[2]);

        String feedbackMessage;
        switch (action) {
            case "like":
                feedbackMessage = "👍 Thank you for your feedback! We're glad the prediction was helpful.";
                sendMessage(chatId, feedbackMessage);
                storeFeedback(contentHash, "like", isFake, null, null);
                // Show language selection after like feedback
                showLanguageSelectionButtons(chatId);
                break;
            case "dislike":
                feedbackMessage = "👎 Thank you for your feedback! We'll use this to improve our predictions.";
                sendMessage(chatId, feedbackMessage);
                storeFeedback(contentHash, "dislike", isFake, null, null);
                // Show language selection after dislike feedback
                showLanguageSelectionButtons(chatId);
                break;
            case "report":
                // Store the report context and ask for reason
                pendingReports.put(chatId, new PendingReport(contentHash, isFake));
                
                feedbackMessage = 
                    "🚨 *Report Content*\n\n" +
                    "Please provide the reason for your report.\n" +
                    "Include any specific details about why you believe this content is misclassified.";
                
                sendMessage(chatId, feedbackMessage);
                break;
            default:
                return;
        }

        // Cancel any existing timer for this chat
        Timer existingTimer = feedbackTimers.remove(chatId);
        if (existingTimer != null) {
            existingTimer.cancel();
        }
    }

    private void sendAnalysisResult(long chatId, String content, boolean isFake, double confidence) {
        String responseText = String.format(
            "📰 *News Analysis Result*\n\n" +
            "Content: %s\n\n" +
            "Prediction: %s\n" +
            "Confidence: %.2f%%\n\n" +
            "Was this prediction helpful?",
            content,
            isFake ? "✅ Likely Real" : "❌ Likely Fake",
            confidence * 100
        );

        SendMessage message = new SendMessage();
        message.setChatId(chatId);
        message.setText(responseText);
        message.enableMarkdown(true);

        // Generate a simple hash of the content for tracking
        String contentHash = String.valueOf(content.hashCode());
        
        // Add feedback buttons
        message.setReplyMarkup(createFeedbackKeyboard(contentHash, isFake));

        try {
            execute(message);
        } catch (TelegramApiException e) {
            e.printStackTrace();
        }

        // Start a timer to show language buttons after 60 seconds if no feedback
        // Only start the timer if there isn't already a pending report
        if (!pendingReports.containsKey(chatId)) {
            Timer timer = new Timer();
            timer.schedule(new TimerTask() {
                @Override
                public void run() {
                    showLanguageSelectionButtons(chatId);
                }
            }, 60000); // 60 seconds
            feedbackTimers.put(chatId, timer);
        }
    }

    private void showLanguageSelectionButtons(long chatId) {
        InlineKeyboardMarkup markup = new InlineKeyboardMarkup();
        List<List<InlineKeyboardButton>> keyboard = new ArrayList<>();
        List<InlineKeyboardButton> row = new ArrayList<>();

        // English button
        InlineKeyboardButton englishButton = new InlineKeyboardButton();
        englishButton.setText("🇬🇧 English");
        englishButton.setCallbackData("lang:en:0");
        row.add(englishButton);

        // Malay button
        InlineKeyboardButton malayButton = new InlineKeyboardButton();
        malayButton.setText("🇲🇾 Malay");
        malayButton.setCallbackData("lang:ms:0");
        row.add(malayButton);

        keyboard.add(row);
        markup.setKeyboard(keyboard);

        SendMessage languageMessage = new SendMessage();
        languageMessage.setChatId(chatId);
        languageMessage.setText("🌐 Please select your preferred language for the next analysis:");
        languageMessage.setReplyMarkup(markup);

        try {
            execute(languageMessage);
        } catch (TelegramApiException e) {
            e.printStackTrace();
        }
    }

    private void sendMessage(long chatId, String text) {
            SendMessage message = new SendMessage();
            message.setChatId(chatId);
        message.setText(text);
        message.enableMarkdown(true);
    
            try {
                execute(message);
            } catch (TelegramApiException e) {
                e.printStackTrace();
            }
        }

    private InlineKeyboardMarkup createFeedbackKeyboard(String contentHash, boolean isFake) {
        InlineKeyboardMarkup markup = new InlineKeyboardMarkup();
        List<List<InlineKeyboardButton>> keyboard = new ArrayList<>();
        List<InlineKeyboardButton> row = new ArrayList<>();

        // Like button
        InlineKeyboardButton likeButton = new InlineKeyboardButton();
        likeButton.setText("👍 Accurate");
        likeButton.setCallbackData("like:" + contentHash + ":" + isFake);
        row.add(likeButton);

        // Dislike button
        InlineKeyboardButton dislikeButton = new InlineKeyboardButton();
        dislikeButton.setText("👎 Inaccurate");
        dislikeButton.setCallbackData("dislike:" + contentHash + ":" + isFake);
        row.add(dislikeButton);

        // Report button
        InlineKeyboardButton reportButton = new InlineKeyboardButton();
        reportButton.setText("🚨 Report");
        reportButton.setCallbackData("report:" + contentHash + ":" + isFake);
        row.add(reportButton);

        keyboard.add(row);
        markup.setKeyboard(keyboard);
        return markup;
    }
}
