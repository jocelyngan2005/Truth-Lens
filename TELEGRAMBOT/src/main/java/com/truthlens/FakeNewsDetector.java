package com.truthlens;

import java.util.Random;

public class FakeNewsDetector {
    private final Random random;

    public FakeNewsDetector() {
        this.random = new Random();
    }

    public PredictionResult predict(String content) {
        // This is a temporary implementation that returns random results
        // It will be replaced with the actual BERT model later
        boolean isFake = random.nextBoolean();
        double confidence = 0.5 + (random.nextDouble() * 0.5); // Random confidence between 0.5 and 1.0
        
        return new PredictionResult(isFake, confidence);
    }
} 