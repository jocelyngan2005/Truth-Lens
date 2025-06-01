package com.truthlens;

import org.telegram.telegrambots.meta.TelegramBotsApi;
import org.telegram.telegrambots.meta.exceptions.TelegramApiException;
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession;

public class TruthLensBot {
    public static void main(String[] args) {
        try {
            String botToken = "8140995207:AAFsiOpulbxmBysC5RqB3eoBwU6RJBmoca4";
            String botUsername = "TruthLensBot";

            TelegramBotsApi botsApi = new TelegramBotsApi(DefaultBotSession.class);
            botsApi.registerBot(new Bot(botToken, botUsername));
            System.out.println("TruthLens Bot started successfully!");
        } catch (TelegramApiException e) {
            e.printStackTrace();
        }
    }
} 