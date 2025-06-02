package com.truthlens;

import org.telegram.telegrambots.meta.TelegramBotsApi;
import org.telegram.telegrambots.meta.exceptions.TelegramApiException;
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession;

public class TruthLensBot {
    public static void main(String[] args) {
        try {
            String botToken = "YOUR_BOT_TOKEN";
            String botUsername = "YOUR_BOT_USERNAME";
            // Replace with your actual bot token and username
          
            TelegramBotsApi botsApi = new TelegramBotsApi(DefaultBotSession.class);
            botsApi.registerBot(new Bot(botToken, botUsername));
            System.out.println("TruthLens Bot started successfully!");
        } catch (TelegramApiException e) {
            e.printStackTrace();
        }
    }
} 
