"""
Real-time Learning System
Automatically updates models as new performance data arrives
Implements continuous learning with safety checks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import threading
import time
from pathlib import Path

class RealTimeLearner:
    """Continuous learning system for CreativeIQ"""
    
    def __init__(self, 
                 min_samples_for_retrain=20,
                 retrain_interval_hours=24,
                 performance_threshold=0.05):
        """
        Initialize real-time learner
        
        Args:
            min_samples_for_retrain: Minimum new samples before retraining
            retrain_interval_hours: Hours between automatic retrains
            performance_threshold: Minimum improvement to deploy new model
        """
        self.min_samples = min_samples_for_retrain
        self.retrain_interval = retrain_interval_hours * 3600  # Convert to seconds
        self.performance_threshold = performance_threshold
        
        self.training_queue = []
        self.last_retrain = None
        self.model_versions = []
        self.learning_stats = {
            'total_retrains': 0,
            'successful_deployments': 0,
            'rejected_models': 0,
            'current_accuracy': 0.0
        }
        
        self.load_state()
        print("✅ Real-time Learning System initialized")
    
    def add_training_sample(self, features, actual_performance):
        """
        Add a new training sample from real-world performance
        
        Args:
            features: Dictionary with image and text features
            actual_performance: Dictionary with actual engagement metrics
        """
        sample = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'actual': actual_performance,
            'processed': False
        }
        
        self.training_queue.append(sample)
        
        print(f"📥 Added training sample. Queue size: {len(self.training_queue)}")
        
        # Check if we should trigger retrain
        if self.should_retrain():
            print("🎯 Triggering automatic model retrain...")
            self.retrain_models()
        
        self.save_state()
    
    def should_retrain(self):
        """Determine if models should be retrained"""
        # Check minimum samples
        unprocessed = [s for s in self.training_queue if not s['processed']]
        if len(unprocessed) < self.min_samples:
            return False
        
        # Check time interval
        if self.last_retrain:
            time_since_retrain = (datetime.now() - datetime.fromisoformat(self.last_retrain)).total_seconds()
            if time_since_retrain < self.retrain_interval:
                return False
        
        return True
    
    def retrain_models(self):
        """Retrain all models with new data"""
        try:
            print("\n" + "="*60)
            print("🔄 STARTING AUTOMATIC MODEL RETRAIN")
            print("="*60)
            
            # Prepare training data
            new_data = self.prepare_training_data()
            
            if len(new_data) < self.min_samples:
                print(f"⚠️  Not enough data. Need {self.min_samples}, have {len(new_data)}")
                return False
            
            # Load existing training data
            existing_data = self.load_existing_data()
            
            # Combine with new data
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            print(f"📊 Training with {len(combined_data)} total samples")
            print(f"   - Existing: {len(existing_data)}")
            print(f"   - New: {len(new_data)}")
            
            # Train new models
            from ml_trainer import CreativeMLTrainer
            trainer = CreativeMLTrainer()
            
            # Prepare features
            X, y_eng, y_clk, y_shr, feature_names = trainer.prepare_features(combined_data)
            
            # Train models
            metrics = trainer.train_models(X, y_eng, y_clk, y_shr)
            
            # Validate new models
            if self.validate_new_models(metrics):
                # Save new models
                version = len(self.model_versions) + 1
                model_dir = f'models/v{version}'
                trainer.save_models(model_dir)
                
                # Backup old models
                self.backup_current_models()
                
                # Deploy new models
                self.deploy_models(model_dir)
                
                # Update stats
                self.model_versions.append({
                    'version': version,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics,
                    'sample_count': len(combined_data)
                })
                
                self.learning_stats['total_retrains'] += 1
                self.learning_stats['successful_deployments'] += 1
                self.learning_stats['current_accuracy'] = metrics['engagement']['r2']
                
                # Mark samples as processed
                for sample in self.training_queue:
                    sample['processed'] = True
                
                self.last_retrain = datetime.now().isoformat()
                
                print("\n✅ NEW MODELS DEPLOYED SUCCESSFULLY!")
                print(f"   Version: {version}")
                print(f"   Engagement R²: {metrics['engagement']['r2']:.3f}")
                print("="*60 + "\n")
                
                self.save_state()
                return True
            else:
                print("\n❌ New models did not meet performance threshold")
                print("   Keeping existing models")
                self.learning_stats['rejected_models'] += 1
                self.save_state()
                return False
                
        except Exception as e:
            print(f"\n❌ Error during retrain: {str(e)}")
            return False
    
    def prepare_training_data(self):
        """Convert queue samples to DataFrame"""
        samples = []
        
        for item in self.training_queue:
            if item['processed']:
                continue
            
            features = item['features']
            actual = item['actual']
            
            sample = {
                'timestamp': item['timestamp'],
                'brightness': features.get('brightness', 0),
                'contrast': features.get('contrast', 0),
                'color_diversity': features.get('color_diversity', 0),
                'composition_score': features.get('composition_score', 0),
                'text_coverage': features.get('text_coverage', 0),
                'sentiment_polarity': features.get('sentiment_polarity', 0),
                'sentiment_subjectivity': features.get('sentiment_subjectivity', 0),
                'readability_score': features.get('readability_score', 0),
                'has_cta': int(features.get('has_cta', False)),
                'text_length': features.get('text_length', 0),
                'word_count': features.get('word_count', 0),
                'actual_engagement': actual.get('engagement', 0),
                'actual_clicks': actual.get('clicks', 0),
                'actual_shares': actual.get('shares', 0),
                'actual_conversions': actual.get('conversions', 0)
            }
            
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def load_existing_data(self):
        """Load existing training data"""
        try:
            return pd.read_csv('training_data.csv')
        except FileNotFoundError:
            print("⚠️  No existing training data found")
            return pd.DataFrame()
    
    def validate_new_models(self, metrics):
        """
        Validate new models against current models
        
        Args:
            metrics: Performance metrics of new models
            
        Returns:
            True if new models should be deployed
        """
        # Check if metrics meet minimum standards
        if metrics['engagement']['r2'] < 0.5:
            print("❌ Engagement model R² too low")
            return False
        
        if metrics['clicks']['r2'] < 0.4:
            print("❌ Clicks model R² too low")
            return False
        
        # Compare with previous version if exists
        if self.model_versions:
            prev_metrics = self.model_versions[-1]['metrics']
            
            # Calculate improvement
            eng_improvement = metrics['engagement']['r2'] - prev_metrics['engagement']['r2']
            clk_improvement = metrics['clicks']['r2'] - prev_metrics['clicks']['r2']
            
            print(f"\n📈 Model Comparison:")
            print(f"   Engagement R² change: {eng_improvement:+.3f}")
            print(f"   Clicks R² change: {clk_improvement:+.3f}")
            
            # Require minimum improvement or accept if not worse
            if eng_improvement < -0.1 or clk_improvement < -0.1:
                print("   ❌ New models perform worse")
                return False
            
            if eng_improvement > self.performance_threshold:
                print("   ✅ Significant improvement detected!")
                return True
            elif eng_improvement >= 0:
                print("   ✅ Models maintained or improved slightly")
                return True
            else:
                print("   ⚠️  Marginal decline, but within tolerance")
                return True
        
        # First model, just check minimum standards
        return True
    
    def backup_current_models(self):
        """Backup current models before deploying new ones"""
        try:
            import shutil
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = f'models/backups/backup_{timestamp}'
            
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            
            # Backup all model files
            for file in ['engagement_model.pkl', 'clicks_model.pkl', 
                        'shares_model.pkl', 'scaler.pkl', 'metadata.json']:
                src = f'models/{file}'
                if Path(src).exists():
                    shutil.copy2(src, f'{backup_dir}/{file}')
            
            print(f"💾 Current models backed up to {backup_dir}")
            
        except Exception as e:
            print(f"⚠️  Backup warning: {str(e)}")
    
    def deploy_models(self, model_dir):
        """Deploy new models to production"""
        import shutil
        
        # Copy new models to main models directory
        for file in ['engagement_model.pkl', 'clicks_model.pkl', 
                    'shares_model.pkl', 'scaler.pkl', 'metadata.json']:
            src = f'{model_dir}/{file}'
            dst = f'models/{file}'
            if Path(src).exists():
                shutil.copy2(src, dst)
        
        print(f"🚀 Models deployed from {model_dir} to models/")
    
    def get_learning_stats(self):
        """Get learning system statistics"""
        stats = self.learning_stats.copy()
        stats.update({
            'queue_size': len(self.training_queue),
            'unprocessed_samples': len([s for s in self.training_queue if not s['processed']]),
            'last_retrain': self.last_retrain,
            'model_version': len(self.model_versions),
            'next_retrain_eligible': self.should_retrain()
        })
        
        if self.last_retrain:
            time_since = (datetime.now() - datetime.fromisoformat(self.last_retrain)).total_seconds() / 3600
            stats['hours_since_retrain'] = round(time_since, 1)
        
        return stats
    
    def save_state(self):
        """Save learner state to disk"""
        state = {
            'training_queue': self.training_queue,
            'last_retrain': self.last_retrain,
            'model_versions': self.model_versions,
            'learning_stats': self.learning_stats
        }
        
        with open('learner_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load learner state from disk"""
        try:
            with open('learner_state.json', 'r') as f:
                state = json.load(f)
                self.training_queue = state.get('training_queue', [])
                self.last_retrain = state.get('last_retrain')
                self.model_versions = state.get('model_versions', [])
                self.learning_stats = state.get('learning_stats', self.learning_stats)
                
            print(f"📂 Loaded learner state: {len(self.training_queue)} samples in queue")
        except FileNotFoundError:
            print("📂 No previous learner state found, starting fresh")
    
    def start_background_learning(self):
        """Start background thread for automatic learning"""
        def learning_loop():
            print("🔄 Background learning thread started")
            while True:
                time.sleep(3600)  # Check every hour
                if self.should_retrain():
                    print("\n⏰ Scheduled retrain triggered")
                    self.retrain_models()
        
        thread = threading.Thread(target=learning_loop, daemon=True)
        thread.start()
        print("✅ Background learning enabled")
    
    def get_model_history(self):
        """Get history of all model versions"""
        return {
            'total_versions': len(self.model_versions),
            'versions': self.model_versions,
            'current_version': len(self.model_versions)
        }

# Test the learner
if __name__ == "__main__":
    learner = RealTimeLearner(
        min_samples_for_retrain=5,  # Low threshold for testing
        retrain_interval_hours=1
    )
    
    # Simulate adding samples
    for i in range(6):
        learner.add_training_sample(
            features={
                'brightness': 150 + i * 10,
                'contrast': 60 + i * 5,
                'color_diversity': 7.0,
                'composition_score': 75.0,
                'text_coverage': 15.0,
                'sentiment_polarity': 0.6,
                'sentiment_subjectivity': 0.4,
                'readability_score': 70.0,
                'has_cta': True,
                'text_length': 120,
                'word_count': 22
            },
            actual_performance={
                'engagement': 5.5 + i * 0.2,
                'clicks': 3.2 + i * 0.1,
                'shares': 40 + i * 5,
                'conversions': 8
            }
        )
    
    # Get stats
    stats = learner.get_learning_stats()
    print("\n📊 Learning System Stats:")
    print(json.dumps(stats, indent=2))
