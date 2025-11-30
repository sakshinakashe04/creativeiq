"""
Machine Learning Trainer
Trains models to predict creative performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from datetime import datetime

class CreativeMLTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def load_training_data(self, filepath='training_data.csv'):
        """Load training data from CSV"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df)} training samples")
            return df
        except FileNotFoundError:
            print("❌ Training data not found. Please collect data first.")
            return None
    
    def prepare_features(self, df):
        """Prepare feature matrix and target variables"""
        # Feature columns
        feature_cols = [
            'brightness', 'contrast', 'color_diversity', 
            'composition_score', 'text_coverage',
            'sentiment_polarity', 'sentiment_subjectivity',
            'readability_score', 'has_cta', 'text_length', 'word_count'
        ]
        
        X = df[feature_cols].fillna(0)
        
        # Target variables
        y_engagement = df['actual_engagement']
        y_clicks = df['actual_clicks']
        y_shares = df['actual_shares']
        
        return X, y_engagement, y_clicks, y_shares, feature_cols
    
    def train_models(self, X, y_engagement, y_clicks, y_shares):
        """Train multiple models for different metrics"""
        
        # Split data
        X_train, X_test, y_eng_train, y_eng_test = train_test_split(
            X, y_engagement, test_size=0.2, random_state=42
        )
        _, _, y_clk_train, y_clk_test = train_test_split(
            X, y_clicks, test_size=0.2, random_state=42
        )
        _, _, y_shr_train, y_shr_test = train_test_split(
            X, y_shares, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        print("\n🎯 Training Models...")
        print("="*60)
        
        # Train Engagement Model
        print("\n📈 Training Engagement Predictor...")
        eng_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        eng_model.fit(X_train_scaled, y_eng_train)
        eng_pred = eng_model.predict(X_test_scaled)
        eng_mae = mean_absolute_error(y_eng_test, eng_pred)
        eng_r2 = r2_score(y_eng_test, eng_pred)
        
        print(f"   MAE: {eng_mae:.3f}")
        print(f"   R² Score: {eng_r2:.3f}")
        
        self.models['engagement'] = eng_model
        
        # Train Clicks Model
        print("\n🖱️  Training Click-Through Predictor...")
        clk_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        clk_model.fit(X_train_scaled, y_clk_train)
        clk_pred = clk_model.predict(X_test_scaled)
        clk_mae = mean_absolute_error(y_clk_test, clk_pred)
        clk_r2 = r2_score(y_clk_test, clk_pred)
        
        print(f"   MAE: {clk_mae:.3f}")
        print(f"   R² Score: {clk_r2:.3f}")
        
        self.models['clicks'] = clk_model
        
        # Train Shares Model
        print("\n🔄 Training Shares Predictor...")
        shr_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        shr_model.fit(X_train_scaled, y_shr_train)
        shr_pred = shr_model.predict(X_test_scaled)
        shr_mae = mean_absolute_error(y_shr_test, shr_pred)
        shr_r2 = r2_score(y_shr_test, shr_pred)
        
        print(f"   MAE: {shr_mae:.3f}")
        print(f"   R² Score: {shr_r2:.3f}")
        
        self.models['shares'] = shr_model
        
        # Calculate feature importance
        self.feature_importance['engagement'] = eng_model.feature_importances_
        self.feature_importance['clicks'] = clk_model.feature_importances_
        self.feature_importance['shares'] = shr_model.feature_importances_
        
        print("\n✅ All models trained successfully!")
        
        return {
            'engagement': {'mae': eng_mae, 'r2': eng_r2},
            'clicks': {'mae': clk_mae, 'r2': clk_r2},
            'shares': {'mae': shr_mae, 'r2': shr_r2}
        }
    
    def save_models(self, model_dir='models'):
        """Save trained models to disk"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.models['engagement'], f'{model_dir}/engagement_model.pkl')
        joblib.dump(self.models['clicks'], f'{model_dir}/clicks_model.pkl')
        joblib.dump(self.models['shares'], f'{model_dir}/shares_model.pkl')
        joblib.dump(self.scalers['main'], f'{model_dir}/scaler.pkl')
        
        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'feature_importance': {
                k: v.tolist() for k, v in self.feature_importance.items()
            }
        }
        
        with open(f'{model_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Models saved to '{model_dir}/' directory")
    
    def show_feature_importance(self, feature_names):
        """Display which features matter most"""
        print("\n🔍 Feature Importance Analysis:")
        print("="*60)
        
        for metric in ['engagement', 'clicks', 'shares']:
            print(f"\n{metric.upper()} Prediction:")
            importance = self.feature_importance[metric]
            
            # Sort by importance
            indices = np.argsort(importance)[::-1]
            
            for i in range(min(5, len(indices))):
                idx = indices[i]
                print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.3f}")

# Main training script
if __name__ == "__main__":
    trainer = CreativeMLTrainer()
    
    # Load data
    df = trainer.load_training_data()
    
    if df is not None and len(df) >= 20:  # Need at least 20 samples
        # Prepare features
        X, y_eng, y_clk, y_shr, feature_names = trainer.prepare_features(df)
        
        # Train models
        metrics = trainer.train_models(X, y_eng, y_clk, y_shr)
        
        # Show feature importance
        trainer.show_feature_importance(feature_names)
        
        # Save models
        trainer.save_models()
        
        print("\n🎉 Training complete! Models are ready to use.")
    else:
        print("\n⚠️  Need at least 20 training samples to train models.")
        print("   Current samples:", len(df) if df is not None else 0)
        print("\n💡 To collect data:")
        print("   1. Use your app normally")
        print("   2. Track actual performance of creatives")
        print("   3. Add samples using collect_training_data.py")
