import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ml_predictor import MLPredictor
    ml_predictor = MLPredictor()
    USE_ML = True
except:
    USE_ML = False
    print("⚠️  ML models not available, using rule-based predictions")



"""Performance Predictor Module
Combines image and text analysis to predict engagement"""


def predict(self, image_analysis, text_analysis):
    # Try ML prediction first
    if USE_ML and ml_predictor.models_loaded:
        ml_result = ml_predictor.predict(image_analysis, text_analysis)
        # Continue with existing logic for improvements
        all_issues = image_analysis.get('issues', []) + text_analysis.get('issues', [])
        improvements = self._generate_improvements(all_issues)
        
        return {
            **ml_result,
            'projected_score': min(100, ml_result['overall_score'] + 10),
            'issues': all_issues,
            'improvements': improvements,
            'improvement_potential': 10.0
        }
    
    # Otherwise use existing rule-based logic
    # ... (keep your existing code)
class PerformancePredictor:
    """Predicts engagement and performance metrics"""
    
    def predict(self, image_analysis, text_analysis):
        """
        Main prediction function
        Args:
            image_analysis: Results from ImageAnalyzer
            text_analysis: Results from TextAnalyzer
        Returns:
            Dictionary with predictions and recommendations
        """
        # Weighted overall score (images matter more)
        image_weight = 0.6
        text_weight = 0.4
        
        overall_score = (
            image_analysis['image_score'] * image_weight +
            text_analysis['text_score'] * text_weight
        )
        
        # Predict engagement rate (2-8% range)
        base_engagement = 2.0
        score_multiplier = overall_score / 100
        predicted_engagement = base_engagement + (score_multiplier * 4)
        
        # Predict click-through rate
        predicted_clicks = predicted_engagement * 0.6
        
        # Predict shares (emotional content gets shared more)
        sentiment_score = text_analysis['sentiment']['score']
        base_shares = 10
        sentiment_boost = (sentiment_score / 100) * 50
        predicted_shares = int(base_shares + sentiment_boost)
        
        # Combine all issues
        all_issues = image_analysis.get('issues', []) + text_analysis.get('issues', [])
        
        # Calculate improvement potential
        total_negative_impact = sum(
            abs(issue['impact']) for issue in all_issues 
            if issue.get('impact', 0) < 0
        )
        
        projected_score = min(100, overall_score + total_negative_impact)
        
        # Generate improvements
        improvements = self._generate_improvements(all_issues)
        
        return {
            'overall_score': round(overall_score, 1),
            'projected_score': round(projected_score, 1),
            'predictions': {
                'engagement': round(predicted_engagement, 2),
                'clicks': round(predicted_clicks, 2),
                'shares': predicted_shares
            },
            'issues': all_issues,
            'improvements': improvements,
            'improvement_potential': round(total_negative_impact, 1)
        }
    
    def _generate_improvements(self, issues):
        """Converts issues into actionable improvements"""
        improvements = []
        
        # Sort by impact (worst first)
        sorted_issues = sorted(issues, key=lambda x: x.get('impact', 0))
        
        for issue in sorted_issues:
            if issue.get('impact', 0) < 0:
                improvement = {
                    'aspect': issue['category'],
                    'current_problem': issue['issue'],
                    'recommended_fix': issue['fix'],
                    'expected_boost': f"+{abs(issue['impact'])}%",
                    'priority': 'HIGH' if abs(issue['impact']) >= 15 else 'MEDIUM' if abs(issue['impact']) >= 10 else 'LOW'
                }
                improvements.append(improvement)
        
        return improvements
