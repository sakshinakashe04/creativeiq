"""
Generate Sample Training Data
Creates realistic sample data for initial training
"""

from collect_training_data import DataCollector
import random

collector = DataCollector()

# Generate 50 sample creatives
print("🎲 Generating sample training data...")

for i in range(50):
    # Random image features
    brightness = random.uniform(80, 220)
    contrast = random.uniform(30, 90)
    
    # Random text features
    has_cta = random.choice([True, False])
    sentiment = random.uniform(-0.3, 0.8)
    text_length = random.randint(50, 250)
    
    # Simulate realistic performance based on features
    # Good features = better performance
    base_engagement = 2.0
    if brightness > 120 and brightness < 180:
        base_engagement += 1.0
    if contrast > 50:
        base_engagement += 0.8
    if has_cta:
        base_engagement += 1.5
    if sentiment > 0.3:
        base_engagement += 1.2
    
    # Add some randomness
    actual_engagement = base_engagement + random.uniform(-0.5, 0.5)
    actual_clicks = actual_engagement * random.uniform(0.5, 0.7)
    actual_shares = int(actual_engagement * random.uniform(5, 15))
    
    collector.add_sample(
        image_features={
            'brightness': brightness,
            'contrast': contrast,
            'color_diversity': random.uniform(3, 9),
            'composition_score': random.uniform(40, 90),
            'text_coverage': random.uniform(5, 25)
        },
        text_features={
            'sentiment_polarity': sentiment,
            'sentiment_subjectivity': random.uniform(0.2, 0.8),
            'readability_score': random.uniform(40, 90),
            'has_cta': has_cta,
            'text_length': text_length,
            'word_count': text_length // 6
        },
        actual_performance={
            'engagement': actual_engagement,
            'clicks': actual_clicks,
            'shares': actual_shares,
            'conversions': int(actual_clicks * random.uniform(0.1, 0.3))
        }
    )

print("\n✅ Generated 50 training samples!")
print("📊 Run 'python3 ml_trainer.py' to train models")
