"""
Text Analyzer Module
Analyzes captions/text for sentiment, readability, tone, and CTA
"""

from textblob import TextBlob
import re

class TextAnalyzer:
    """Analyzes caption/text content for engagement prediction"""
    
    def __init__(self):
        self.cta_keywords = [
            'click', 'shop', 'buy', 'get', 'download', 'subscribe',
            'sign up', 'join', 'learn', 'discover', 'register', 'try',
            'order', 'save', 'book', 'reserve', 'start', 'explore'
        ]
        
    def analyze(self, text):
        """
        Main text analysis function
        Args:
            text: Caption or text content (string)
        Returns:
            Dictionary with all analysis results
        """
        if not text or len(text.strip()) == 0:
            return self._empty_analysis()
        
        try:
            sentiment = self._analyze_sentiment(text)
            readability = self._analyze_readability(text)
            tone = self._analyze_tone(text)
            cta = self._detect_cta(text)
            length = self._analyze_length(text)
            
            score = self._calculate_text_score(sentiment, readability, tone, cta, length)
            issues = self._identify_text_issues(readability, tone, cta, length)
            
            return {
                'sentiment': sentiment,
                'readability': readability,
                'tone': tone,
                'has_cta': cta,
                'length_analysis': length,
                'text_score': round(score, 1),
                'issues': issues
            }
        except Exception as e:
            print(f"Error in text analysis: {str(e)}")
            return self._empty_analysis()
    
    def _analyze_sentiment(self, text):
        """Analyzes emotional sentiment"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.3:
                emotion = 'positive'
            elif polarity < -0.3:
                emotion = 'negative'
            else:
                emotion = 'neutral'
            
            return {
                'polarity': float(polarity),
                'subjectivity': float(subjectivity),
                'emotion': emotion,
                'score': float((polarity + 1) * 50)
            }
        except:
            return {'polarity': 0, 'subjectivity': 0, 'emotion': 'neutral', 'score': 50}
    
    def _analyze_readability(self, text):
        """Calculates Flesch Reading Ease Score"""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        if not sentences or not words:
            return {'score': 50, 'level': 'easy', 'grade_level': '8th-9th grade'}
        
        syllables = sum(self._count_syllables(word) for word in words)
        
        avg_words = len(words) / len(sentences)
        avg_syllables = syllables / len(words) if words else 0
        
        score = 206.835 - 1.015 * avg_words - 84.6 * avg_syllables
        score = max(0, min(100, score))
        
        if score >= 80:
            level = 'very_easy'
        elif score >= 60:
            level = 'easy'
        elif score >= 30:
            level = 'difficult'
        else:
            level = 'very_difficult'
        
        return {
            'score': float(score),
            'level': level,
            'grade_level': self._score_to_grade(score)
        }
    
    def _analyze_tone(self, text):
        """Detects if tone is formal or casual"""
        formal_indicators = [
            'hereby', 'therefore', 'thus', 'cordially', 'furthermore',
            'nevertheless', 'moreover', 'consequently'
        ]
        casual_indicators = [
            'hey', 'yeah', 'cool', 'awesome', 'wow', 'super', 'amazing',
            'literally', 'totally', 'basically', '!'
        ]
        
        text_lower = text.lower()
        
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        casual_count = sum(1 for word in casual_indicators if word in text_lower)
        
        # Count emojis
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿]', text))
        casual_count += emoji_count
        
        if formal_count > casual_count:
            tone_type = 'formal'
            score = min(100, 60 + formal_count * 10)
        elif casual_count > formal_count:
            tone_type = 'casual'
            score = max(0, 40 - casual_count * 5)
        else:
            tone_type = 'neutral'
            score = 50
        
        return {
            'type': tone_type,
            'formality_score': float(score),
            'emoji_count': emoji_count
        }
    
    def _detect_cta(self, text):
        """Detects call-to-action"""
        text_lower = text.lower()
        
        found_ctas = [cta for cta in self.cta_keywords if cta in text_lower]
        has_link = bool(re.search(r'http[s]?://|www\.', text))
        
        if len(found_ctas) > 1 or has_link:
            strength = 'strong'
        elif len(found_ctas) == 1:
            strength = 'weak'
        else:
            strength = 'none'
        
        return {
            'present': len(found_ctas) > 0 or has_link,
            'keywords_found': found_ctas,
            'has_link': has_link,
            'strength': strength
        }
    
    def _analyze_length(self, text):
        """Checks if text length is optimal"""
        words = len(text.split())
        chars = len(text)
        
        if chars < 50:
            assessment = 'too_short'
        elif chars > 200:
            assessment = 'too_long'
        else:
            assessment = 'optimal'
        
        return {
            'words': words,
            'characters': chars,
            'assessment': assessment
        }
    
    def _calculate_text_score(self, sentiment, readability, tone, cta, length):
        """Calculates overall text quality score"""
        score = 0
        
        score += sentiment['score'] * 0.2
        score += min(25, readability['score'] * 0.25)
        
        if cta['present']:
            score += 20
        
        if length['assessment'] == 'optimal':
            score += 20
        else:
            score += 10
        
        if tone['type'] == 'casual':
            score += 15
        elif tone['type'] == 'neutral':
            score += 10
        else:
            score += 5
        
        return min(100, score)
    
    def _identify_text_issues(self, readability, tone, cta, length):
        """Identifies text issues"""
        issues = []
        
        if readability['level'] in ['difficult', 'very_difficult']:
            issues.append({
                'type': 'warning',
                'category': 'Readability',
                'issue': 'Text is difficult to read',
                'impact': -10,
                'fix': 'Simplify language and use shorter sentences',
                'technical': f'Reading ease: {readability["score"]:.1f} → Target: 60+'
            })
        
        if tone['type'] == 'formal' and tone['formality_score'] > 70:
            issues.append({
                'type': 'warning',
                'category': 'Tone',
                'issue': 'Language too formal for social media',
                'impact': -12,
                'fix': 'Use more conversational, friendly language',
                'technical': f'Formality: {tone["formality_score"]:.0f}/100 → Target: 30-50/100'
            })
        
        if not cta['present']:
            issues.append({
                'type': 'error',
                'category': 'Call-to-Action',
                'issue': 'No clear call-to-action detected',
                'impact': -20,
                'fix': 'Add clear CTA like "Shop Now", "Learn More", or "Sign Up"',
                'technical': 'Include action verbs and links'
            })
        
        if length['assessment'] == 'too_short':
            issues.append({
                'type': 'warning',
                'category': 'Length',
                'issue': 'Caption is very short',
                'impact': -8,
                'fix': 'Expand to 50-150 characters for better engagement',
                'technical': f'Current: {length["characters"]} chars → Target: 50-150'
            })
        
        return issues
    
    def _count_syllables(self, word):
        """Counts syllables in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _score_to_grade(self, score):
        """Converts readability score to grade level"""
        if score >= 90:
            return '5th grade'
        elif score >= 80:
            return '6th grade'
        elif score >= 70:
            return '7th grade'
        elif score >= 60:
            return '8th-9th grade'
        elif score >= 50:
            return '10th-12th grade'
        else:
            return 'College level'
    
    def _empty_analysis(self):
        """Returns default analysis when no text provided"""
        return {
            'sentiment': {'polarity': 0, 'subjectivity': 0, 'emotion': 'neutral', 'score': 50},
            'readability': {'score': 50, 'level': 'easy', 'grade_level': '8th-9th grade'},
            'tone': {'type': 'neutral', 'formality_score': 50, 'emoji_count': 0},
            'has_cta': {'present': False, 'keywords_found': [], 'has_link': False, 'strength': 'none'},
            'length_analysis': {'words': 0, 'characters': 0, 'assessment': 'too_short'},
            'text_score': 0,
            'issues': []
        }