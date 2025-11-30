"""
Advanced NLP Analyzer using BERT and Transformers
Provides sophisticated text analysis
"""

from transformers import pipeline
import torch
import re
from collections import Counter

class AdvancedNLPAnalyzer:
    """Advanced NLP analysis using transformer models"""
    
    def __init__(self):
        print("🔄 Loading NLP models...")
        
        try:
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            print("✅ NLP models loaded!")
            self.models_loaded = True
            
        except Exception as e:
            print(f"⚠️  Using rule-based NLP: {str(e)}")
            self.models_loaded = False
    
    def analyze(self, text):
        """Comprehensive NLP analysis"""
        if not text or len(text.strip()) == 0:
            return self.get_default_analysis()
        
        try:
            sentiment = self.analyze_sentiment(text)
            emotions = self.analyze_emotions(text)
            intent = self.classify_intent(text)
            persuasiveness = self.analyze_persuasiveness(text)
            urgency = self.detect_urgency(text)
            hooks = self.detect_hooks(text)
            power_words = self.detect_power_words(text)
            
            return {
                'sentiment': sentiment,
                'emotions': emotions,
                'intent': intent,
                'persuasiveness': persuasiveness,
                'urgency': urgency,
                'attention_hooks': hooks,
                'power_words': power_words,
                'nlp_score': self.calculate_nlp_score(
                    sentiment, emotions, persuasiveness, hooks
                )
            }
            
        except Exception as e:
            print(f"❌ NLP error: {str(e)}")
            return self.get_default_analysis()
    
    def analyze_sentiment(self, text):
        """Deep sentiment analysis"""
        try:
            if self.models_loaded:
                result = self.sentiment_analyzer(text[:512])[0]
                label = result['label'].lower()
                score = result['score']
                polarity = score if label == 'positive' else -score
                
                return {
                    'label': label,
                    'polarity': round(polarity, 3),
                    'confidence': round(score, 3),
                    'engagement_impact': 'high' if abs(polarity) > 0.7 else 'medium'
                }
            else:
                return self.simple_sentiment(text)
        except:
            return self.simple_sentiment(text)
    
    def simple_sentiment(self, text):
        """Fallback sentiment analysis"""
        positive = ['great', 'amazing', 'love', 'best', 'awesome', 'fantastic']
        negative = ['bad', 'worst', 'hate', 'terrible', 'awful']
        
        text_lower = text.lower()
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        
        if pos > neg:
            return {'label': 'positive', 'polarity': 0.6, 'confidence': 0.7, 'engagement_impact': 'high'}
        elif neg > pos:
            return {'label': 'negative', 'polarity': -0.6, 'confidence': 0.7, 'engagement_impact': 'low'}
        else:
            return {'label': 'neutral', 'polarity': 0.0, 'confidence': 0.5, 'engagement_impact': 'medium'}
    
    def analyze_emotions(self, text):
        """Detect emotions in text"""
        emotion_keywords = {
            'joy': ['happy', 'excited', 'love', 'amazing', 'wonderful'],
            'trust': ['trust', 'reliable', 'honest', 'quality', 'guaranteed'],
            'fear': ['limited', 'hurry', "don't miss", 'last chance', 'urgent'],
            'anticipation': ['coming', 'soon', 'new', 'exclusive', 'first']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for k in keywords if k in text_lower)
            if score > 0:
                scores[emotion] = score / len(keywords)
        
        if scores:
            dominant = max(scores.items(), key=lambda x: x[1])
            return {
                'dominant_emotion': dominant[0],
                'confidence': min(0.9, dominant[1]),
                'all_emotions': [{'emotion': e, 'score': round(s, 2)} for e, s in scores.items()],
                'emotional_diversity': len(scores)
            }
        
        return {'dominant_emotion': 'neutral', 'confidence': 0.5, 'all_emotions': [], 'emotional_diversity': 0}
    
    def classify_intent(self, text):
        """Classify marketing intent"""
        intents = {
            'sale': ['sale', 'discount', 'off', 'save', 'deal', 'promo'],
            'awareness': ['discover', 'learn', 'explore', 'find out', 'see'],
            'conversion': ['buy', 'shop', 'order', 'get', 'purchase', 'subscribe'],
            'engagement': ['share', 'comment', 'join', 'follow', 'like'],
            'urgency': ['now', 'today', 'limited', 'hurry', 'last chance']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for intent, keywords in intents.items():
            score = sum(1 for k in keywords if k in text_lower)
            if score > 0:
                scores[intent] = score
        
        if scores:
            primary = max(scores.items(), key=lambda x: x[1])
            return {
                'primary_intent': primary[0],
                'confidence': min(0.95, primary[1] / 3),
                'all_intents': scores,
                'clarity': 'clear' if primary[1] >= 2 else 'moderate'
            }
        
        return {'primary_intent': 'informational', 'confidence': 0.5, 'all_intents': {}, 'clarity': 'unclear'}
    
    def analyze_persuasiveness(self, text):
        """Analyze persuasive techniques"""
        techniques = {
            'social_proof': ['people', 'customers', 'users', 'everyone', 'millions'],
            'scarcity': ['limited', 'exclusive', 'only', 'rare', 'few left'],
            'urgency': ['now', 'today', 'hurry', 'quick', 'immediate'],
            'authority': ['expert', 'proven', 'research', 'certified', 'award'],
            'reciprocity': ['free', 'gift', 'bonus', 'complimentary'],
            'commitment': ['guarantee', 'promise', 'ensure', 'commitment']
        }
        
        text_lower = text.lower()
        found = {}
        
        for technique, keywords in techniques.items():
            matches = [k for k in keywords if k in text_lower]
            if matches:
                found[technique] = {'count': len(matches), 'keywords': matches}
        
        score = min(100, len(found) * 20 + sum(t['count'] for t in found.values()) * 5)
        
        return {
            'persuasion_score': score,
            'techniques_used': found,
            'technique_count': len(found),
            'effectiveness': 'high' if score > 70 else 'medium' if score > 40 else 'low'
        }
    
    def detect_urgency(self, text):
        """Detect urgency signals"""
        urgency_words = [
            'now', 'today', 'hurry', 'quick', 'fast', 'immediate', 
            'limited', 'ending', 'expires', 'deadline', 'last chance',
            'don\'t miss', 'act now', 'while supplies last'
        ]
        
        text_lower = text.lower()
        found = [w for w in urgency_words if w in text_lower]
        
        urgency_level = min(100, len(found) * 25)
        
        return {
            'urgency_score': urgency_level,
            'urgency_words': found,
            'level': 'high' if urgency_level > 75 else 'medium' if urgency_level > 40 else 'low',
            'has_deadline': any(w in text_lower for w in ['today', 'tonight', 'expires', 'ends'])
        }
    
    def detect_hooks(self, text):
        """Detect attention-grabbing hooks"""
        hooks = {
            'question': bool(re.search(r'\?', text)),
            'exclamation': bool(re.search(r'!', text)),
            'numbers': bool(re.search(r'\d+', text)),
            'emoji': bool(re.search(r'[😀-🙏🌀-🗿]', text)),
            'capitalization': bool(re.search(r'\b[A-Z]{2,}\b', text)),
            'personal': any(word in text.lower() for word in ['you', 'your', 'yours'])
        }
        
        hook_count = sum(hooks.values())
        
        return {
            'hooks_detected': hooks,
            'hook_count': hook_count,
            'attention_score': min(100, hook_count * 15),
            'engagement_potential': 'high' if hook_count >= 4 else 'medium' if hook_count >= 2 else 'low'
        }
    
    def detect_power_words(self, text):
        """Detect high-impact power words"""
        power_categories = {
            'value': ['free', 'guarantee', 'proven', 'results', 'save', 'value'],
            'emotion': ['love', 'hate', 'fear', 'dream', 'hope', 'amazing'],
            'action': ['get', 'start', 'transform', 'discover', 'unlock', 'achieve'],
            'exclusive': ['exclusive', 'limited', 'secret', 'insider', 'vip', 'special']
        }
        
        text_lower = text.lower()
        found_words = {}
        
        for category, words in power_categories.items():
            matches = [w for w in words if w in text_lower]
            if matches:
                found_words[category] = matches
        
        total_count = sum(len(w) for w in found_words.values())
        
        return {
            'power_words': found_words,
            'total_count': total_count,
            'impact_score': min(100, total_count * 10),
            'effectiveness': 'high' if total_count >= 5 else 'medium' if total_count >= 2 else 'low'
        }
    
    def calculate_nlp_score(self, sentiment, emotions, persuasiveness, hooks):
        """Calculate overall NLP quality score"""
        score = 0
        
        # Sentiment contribution (20 points)
        if sentiment['engagement_impact'] == 'high':
            score += 20
        else:
            score += 10
        
        # Emotional appeal (25 points)
        score += min(25, emotions['emotional_diversity'] * 8)
        
        # Persuasiveness (30 points)
        score += persuasiveness['persuasion_score'] * 0.3
        
        # Hooks (25 points)
        score += hooks['attention_score'] * 0.25
        
        return round(min(100, score), 1)
    
    def get_default_analysis(self):
        """Return default analysis"""
        return {
            'sentiment': {'label': 'neutral', 'polarity': 0, 'confidence': 0.5, 'engagement_impact': 'medium'},
            'emotions': {'dominant_emotion': 'neutral', 'confidence': 0.5, 'all_emotions': [], 'emotional_diversity': 0},
            'intent': {'primary_intent': 'informational', 'confidence': 0.5, 'all_intents': {}, 'clarity': 'unclear'},
            'persuasiveness': {'persuasion_score': 0, 'techniques_used': {}, 'technique_count': 0, 'effectiveness': 'low'},
            'urgency': {'urgency_score': 0, 'urgency_words': [], 'level': 'low', 'has_deadline': False},
            'attention_hooks': {'hooks_detected': {}, 'hook_count': 0, 'attention_score': 0, 'engagement_potential': 'low'},
            'power_words': {'power_words': {}, 'total_count': 0, 'impact_score': 0, 'effectiveness': 'low'},
            'nlp_score': 50.0
        }

# Test
if __name__ == "__main__":
    analyzer = AdvancedNLPAnalyzer()
    
    test_text = "🔥 Limited time offer! Get 50% OFF today only. Don't miss out on this amazing deal!"
    result = analyzer.analyze(test_text)
    
    print("\n📊 NLP Analysis Results:")
    print(f"Sentiment: {result['sentiment']['label']}")
    print(f"Intent: {result['intent']['primary_intent']}")
    print(f"Persuasion Score: {result['persuasiveness']['persuasion_score']}")
    print(f"Overall NLP Score: {result['nlp_score']}")
