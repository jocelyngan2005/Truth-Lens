package com.truthlens;

public class PredictionResult {
    private final boolean isFake;
    private final double confidence;

    public PredictionResult(boolean isFake, double confidence) {
        this.isFake = isFake;
        this.confidence = confidence;
    }

    public boolean isFake() {
        return isFake;
    }

    public double getConfidence() {
        return confidence;
    }
} 