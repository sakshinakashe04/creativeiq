"""
Training Data Collection
Collects creative performance data for ML training
"""

import pandas as pd
import json
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.data_file = 'training_data.csv'
        
    def add_sample(self, image_features, text_features, actual_performance):
        """
        Add a training sample
        
        Args:
            image_features: dict with brightness, contrast, etc.
            text_features: dict with sentiment, readability, etc.
            actual_performance: dict with actual engagement, clicks, shares
        """
        sample = {
            'timestamp': datetime.now().isoformat(),
            # Image features
            'brightness': image_features.get('brightness', 0),
            'contrast': image_features.get('contrast', 0),
            'color_diversity': image_features.get('color_diversity', 0),
            'composition_score': image_features.get('composition_score', 0),
            'text_coverage': image_features.get('text_coverage', 0),
            
            # Text features
            'sentiment_polarity': text_features.get('sentiment_polarity', 0),
            'sentiment_subjectivity': text_features.get('sentiment_subjectivity', 0),
            'readability_score': text_features.get('readability_score', 0),
            'has_cta': int(text_features.get('has_cta', False)),
            'text_length': text_features.get('text_length', 0),
            'word_count': text_features.get('word_count', 0),
            
            # Actual performance (what we want to predict)
            'actual_engagement': actual_performance.get('engagement', 0),
            'actual_clicks': actual_performance.get('clicks', 0),
            'actual_shares': actual_performance.get('shares', 0),
            'actual_conversions': actual_performance.get('conversions', 0)
        }
        
        # Append to CSV
        df = pd.DataFrame([sample])
        try:
            existing_df = pd.read_csv(self.data_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_csv(self.data_file, index=False)
        print(f"✅ Added training sample. Total samples: {len(df)}")
        
    def load_data(self):
        """Load all training data"""
        try:
            return pd.read_csv(self.data_file)
        except FileNotFoundError:
            print("❌ No training data found yet")
            return None
    
    def get_stats(self):
        """Get statistics about training data"""
        df = self.load_data()
        if df is not None:
            return {
                'total_samples': len(df),
                'avg_engagement': df['actual_engagement'].mean(),
                'avg_clicks': df['actual_clicks'].mean(),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            }
        return None

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    
    # Example: Add a sample
    collector.add_sample(
        image_features={
            'brightness': 145.5,
            'contrast': 65.2,
            'color_diversity': 7.5,
            'composition_score': 78.0,
            'text_coverage': 12.3
        },
        text_features={
            'sentiment_polarity': 0.6,
            'sentiment_subjectivity': 0.4,
            'readability_score': 75.0,
            'has_cta': True,
            'text_length': 145,
            'word_count': 28
        },
        actual_performance={
            'engagement': 5.8,  # 5.8% engagement rate
            'clicks': 3.2,      # 3.2% CTR
            'shares': 45,       # 45 shares
            'conversions': 12   # 12 conversions
        }
    )
    
    # Show stats
    stats = collector.get_stats()
    print(f"\n📊 Training Data Stats:")
    print(json.dumps(stats, indent=2))
