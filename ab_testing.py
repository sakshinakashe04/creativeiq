"""
A/B Testing Framework
Compare multiple creative variations and determine winners
"""

import json
import pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats

class ABTestingFramework:
    """Framework for comparing creative variations"""
    
    def __init__(self):
        self.tests = {}
        self.test_file = 'ab_tests.json'
        self.load_tests()
    
    def create_test(self, test_name, variants):
        """
        Create a new A/B test
        
        Args:
            test_name: Name of the test
            variants: List of variant names (e.g., ['A', 'B', 'C'])
        """
        test_id = f"test_{len(self.tests) + 1}_{int(datetime.now().timestamp())}"
        
        self.tests[test_id] = {
            'name': test_name,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'variants': {
                variant: {
                    'predictions': {},
                    'actual_performance': {},
                    'sample_size': 0,
                    'winner_probability': 0
                } for variant in variants
            },
            'winner': None,
            'confidence': 0
        }
        
        self.save_tests()
        print(f"✅ Created A/B test: {test_name} (ID: {test_id})")
        return test_id
    
    def add_variant_data(self, test_id, variant_name, predictions, actual_performance=None):
        """
        Add data for a variant
        
        Args:
            test_id: ID of the test
            variant_name: Name of the variant
            predictions: Predicted metrics
            actual_performance: Actual performance metrics (optional)
        """
        if test_id not in self.tests:
            print(f"❌ Test {test_id} not found")
            return
        
        test = self.tests[test_id]
        
        if variant_name not in test['variants']:
            print(f"❌ Variant {variant_name} not found in test")
            return
        
        variant = test['variants'][variant_name]
        variant['predictions'] = predictions
        
        if actual_performance:
            variant['actual_performance'] = actual_performance
            variant['sample_size'] += 1
        
        self.save_tests()
        print(f"✅ Added data for variant {variant_name}")
    
    def compare_variants(self, test_id):
        """
        Compare all variants and determine winner
        
        Args:
            test_id: ID of the test
            
        Returns:
            Comparison results with winner
        """
        if test_id not in self.tests:
            return {'error': 'Test not found'}
        
        test = self.tests[test_id]
        variants = test['variants']
        
        # Compare predictions
        comparison = {
            'test_name': test['name'],
            'variants': {},
            'winner': None,
            'confidence': 0,
            'recommendation': ''
        }
        
        # Calculate scores for each variant
        for variant_name, data in variants.items():
            pred = data['predictions']
            actual = data['actual_performance']
            
            # Predicted performance score
            predicted_score = (
                pred.get('engagement', 0) * 40 +
                pred.get('clicks', 0) * 30 +
                pred.get('shares', 0) / 10 * 30
            )
            
            # Actual performance score (if available)
            if actual:
                actual_score = (
                    actual.get('engagement', 0) * 40 +
                    actual.get('clicks', 0) * 30 +
                    actual.get('shares', 0) / 10 * 30
                )
            else:
                actual_score = None
            
            comparison['variants'][variant_name] = {
                'predicted_score': round(predicted_score, 2),
                'actual_score': round(actual_score, 2) if actual_score else None,
                'predictions': pred,
                'actual': actual if actual else None,
                'sample_size': data['sample_size']
            }
        
        # Determine winner based on predictions
        winner = max(
            comparison['variants'].items(),
            key=lambda x: x[1]['predicted_score']
        )
        
        comparison['winner'] = winner[0]
        comparison['confidence'] = self.calculate_confidence(variants, winner[0])
        
        # Generate recommendation
        if comparison['confidence'] > 0.95:
            comparison['recommendation'] = f"High confidence winner: Variant {winner[0]}. Deploy immediately."
        elif comparison['confidence'] > 0.80:
            comparison['recommendation'] = f"Moderate confidence winner: Variant {winner[0]}. Consider testing with real traffic."
        else:
            comparison['recommendation'] = "Low confidence. Variants are too similar. Consider more distinctive variations."
        
        # Update test with winner
        self.tests[test_id]['winner'] = winner[0]
        self.tests[test_id]['confidence'] = comparison['confidence']
        self.save_tests()
        
        return comparison
    
    def calculate_confidence(self, variants, winner_name):
        """Calculate statistical confidence in winner"""
        winner_score = variants[winner_name]['predictions'].get('engagement', 0)
        
        other_scores = [
            v['predictions'].get('engagement', 0)
            for name, v in variants.items() if name != winner_name
        ]
        
        if not other_scores:
            return 0.5
        
        # Calculate difference percentage
        avg_other = np.mean(other_scores)
        if avg_other == 0:
            return 0.5
        
        improvement = (winner_score - avg_other) / avg_other
        
        # Convert improvement to confidence
        if improvement > 0.50:  # 50% better
            return 0.99
        elif improvement > 0.30:  # 30% better
            return 0.95
        elif improvement > 0.15:  # 15% better
            return 0.85
        elif improvement > 0.05:  # 5% better
            return 0.70
        else:
            return 0.50
    
    def get_test_report(self, test_id):
        """Generate comprehensive test report"""
        if test_id not in self.tests:
            return {'error': 'Test not found'}
        
        test = self.tests[test_id]
        comparison = self.compare_variants(test_id)
        
        report = {
            'test_id': test_id,
            'test_name': test['name'],
            'status': test['status'],
            'created_at': test['created_at'],
            'winner': comparison['winner'],
            'confidence': comparison['confidence'],
            'recommendation': comparison['recommendation'],
            'variants_summary': []
        }
        
        for variant_name, data in comparison['variants'].items():
            is_winner = variant_name == comparison['winner']
            
            summary = {
                'variant': variant_name,
                'is_winner': is_winner,
                'predicted_score': data['predicted_score'],
                'predicted_engagement': data['predictions'].get('engagement', 0),
                'predicted_clicks': data['predictions'].get('clicks', 0),
                'predicted_shares': data['predictions'].get('shares', 0)
            }
            
            if data['actual']:
                summary['actual_engagement'] = data['actual'].get('engagement', 0)
                summary['actual_clicks'] = data['actual'].get('clicks', 0)
                summary['sample_size'] = data['sample_size']
            
            report['variants_summary'].append(summary)
        
        return report
    
    def save_tests(self):
        """Save tests to file"""
        with open(self.test_file, 'w') as f:
            json.dump(self.tests, f, indent=2)
    
    def load_tests(self):
        """Load tests from file"""
        try:
            with open(self.test_file, 'r') as f:
                self.tests = json.load(f)
        except FileNotFoundError:
            self.tests = {}
    
    def list_tests(self):
        """List all tests"""
        return [
            {
                'test_id': tid,
                'name': t['name'],
                'status': t['status'],
                'variants': list(t['variants'].keys()),
                'winner': t.get('winner'),
                'confidence': t.get('confidence', 0)
            }
            for tid, t in self.tests.items()
        ]

# Example usage
if __name__ == "__main__":
    ab = ABTestingFramework()
    
    # Create a test
    test_id = ab.create_test(
        "Summer Sale Banner Test",
        variants=['A', 'B', 'C']
    )
    
    # Add variant data
    ab.add_variant_data(test_id, 'A', {
        'engagement': 4.5,
        'clicks': 2.7,
        'shares': 35
    })
    
    ab.add_variant_data(test_id, 'B', {
        'engagement': 5.8,
        'clicks': 3.2,
        'shares': 48
    })
    
    ab.add_variant_data(test_id, 'C', {
        'engagement': 4.9,
        'clicks': 2.9,
        'shares': 41
    })
    
    # Compare and get report
    report = ab.get_test_report(test_id)
    
    print("\n📊 A/B Test Report:")
    print(f"Test: {report['test_name']}")
    print(f"Winner: Variant {report['winner']}")
    print(f"Confidence: {report['confidence']:.1%}")
    print(f"\n{report['recommendation']}")
