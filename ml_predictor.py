"""
ML-Powered Predictor
Uses trained models for predictions
"""

import joblib
import numpy as np
import os

class MLPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.engagement_model = joblib.load(f'{self.model_dir}/engagement_model.pkl')
            self.clicks_model = joblib.load(f'{self.model_dir}/clicks_model.pkl')
            self.shares_model = joblib.load(f'{self.model_dir}/shares_model.pkl')
            self.scaler = joblib.load(f'{self.model_dir}/scaler.pkl')
            self.models_loaded = True
            print("✅ ML models loaded successfully")
        except FileNotFoundError:
            print("⚠️  ML models not found. Using rule-based predictions.")
            self.models_loaded = False
    
    def predict(self, image_analysis, text_analysis):
        """
        Make ML-powered predictions
        
        Args:
            image_analysis: Results from ImageAnalyzer
            text_analysis: Results from TextAnalyzer
        
        Returns:
            Dictionary with ML predictions
        """
        if not self.models_loaded:
            return self._fallback_prediction(image_analysis, text_analysis)
        
        # Extract features in same order as training
        features = np.array([[
            image_analysis['brightness']['value'],
            image_analysis['contrast']['value'],
            image_analysis['dominant_colors']['diversity'],
            image_analysis['composition_score']['score'],
            image_analysis['text_readability']['coverage'],
            text_analysis['sentiment']['polarity'],
            text_analysis['sentiment']['subjectivity'],
            text_analysis['readability']['score'],
            int(text_analysis['has_cta']['present']),
            text_analysis['length_analysis']['characters'],
            text_analysis['length_analysis']['words']
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        engagement = float(self.engagement_model.predict(features_scaled)[0])
        clicks = float(self.clicks_model.predict(features_scaled)[0])
        shares = int(self.shares_model.predict(features_scaled)[0])
        
        # Ensure predictions are within reasonable bounds
        engagement = max(1.0, min(15.0, engagement))
        clicks = max(0.5, min(10.0, clicks))
        shares = max(0, shares)
        
        # Calculate overall score based on ML predictions
        overall_score = (
            (engagement / 15.0) * 40 +  # Engagement worth 40 points
            (clicks / 10.0) * 30 +       # Clicks worth 30 points
            min(shares / 100, 1.0) * 30  # Shares worth 30 points
        )
        
        return {
            'ml_powered': True,
            'predictions': {
                'engagement': round(engagement, 2),
                'clicks': round(clicks, 2),
                'shares': shares
            },
            'overall_score': round(overall_score, 1),
            'confidence': 'high' if self.models_loaded else 'low'
        }
    
    def _fallback_prediction(self, image_analysis, text_analysis):
        """Fallback to rule-based when ML models unavailable"""
        # Use original rule-based logic
        image_weight = 0.6
        text_weight = 0.4
        
        overall_score = (
            image_analysis['image_score'] * image_weight +
            text_analysis['text_score'] * text_weight
        )
        
        base_engagement = 2.0
        score_multiplier = overall_score / 100
        predicted_engagement = base_engagement + (score_multiplier * 4)
        predicted_clicks = predicted_engagement * 0.6
        
        sentiment_score = text_analysis['sentiment']['score']
        base_shares = 10
        sentiment_boost = (sentiment_score / 100) * 50
        predicted_shares = int(base_shares + sentiment_boost)
        
        return {
            'ml_powered': False,
            'predictions': {
                'engagement': round(predicted_engagement, 2),
                'clicks': round(predicted_clicks, 2),
                'shares': predicted_shares
            },
            'overall_score': round(overall_score, 1),
            'confidence': 'medium'
        }

# Test the predictor
if __name__ == "__main__":
    predictor = MLPredictor()
    print("ML Predictor ready!")
