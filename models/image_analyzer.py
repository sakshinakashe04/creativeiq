"""
Image Analyzer Module
Analyzes images for brightness, contrast, colors, and composition
"""

import cv2
import numpy as np
from PIL import Image

class ImageAnalyzer:
    """Analyzes creative images for performance prediction"""
    
    def __init__(self):
        self.min_text_size = 14
        self.ideal_contrast = 4.5
        
    def analyze(self, image_file):
        """
        Main analysis function
        Args:
            image_file: Uploaded image file from Flask
        Returns:
            Dictionary with all analysis results
        """
        try:
            # Convert to numpy array
            image = Image.open(image_file)
            img_array = np.array(image)
            
            # Ensure RGB format
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Run all analyses
            brightness = self._analyze_brightness(img_array)
            contrast = self._analyze_contrast(img_array)
            colors = self._analyze_colors(img_array)
            composition = self._analyze_composition(img_array)
            text_metrics = self._analyze_text_readability(img_array)
            
            # Calculate overall score
            score = self._calculate_image_score(
                brightness, contrast, colors, composition, text_metrics
            )
            
            # Identify issues
            issues = self._identify_issues(brightness, contrast, text_metrics)
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'dominant_colors': colors,
                'composition_score': composition,
                'text_readability': text_metrics,
                'image_score': round(score, 1),
                'issues': issues
            }
        except Exception as e:
            print(f"Error in image analysis: {str(e)}")
            # Return default values on error
            return {
                'brightness': {'value': 128, 'status': 'unknown', 'percentage': 50},
                'contrast': {'value': 50, 'status': 'medium'},
                'dominant_colors': {'palette': ['#808080'], 'diversity': 5},
                'composition_score': {'score': 50, 'balance': 'unknown'},
                'text_readability': {'coverage': 5, 'estimated_size': 'adequate'},
                'image_score': 50,
                'issues': []
            }
    
    def _analyze_brightness(self, img):
        """Analyzes image brightness"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 80:
            status = 'too_dark'
        elif brightness > 200:
            status = 'too_bright'
        else:
            status = 'optimal'
            
        return {
            'value': float(brightness),
            'status': status,
            'percentage': float(brightness / 255 * 100)
        }
    
    def _analyze_contrast(self, img):
        """Analyzes contrast ratio"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        
        if contrast < 30:
            status = 'low'
        elif contrast < 60:
            status = 'medium'
        else:
            status = 'high'
            
        return {
            'value': float(contrast),
            'ratio': float(contrast / 128),
            'status': status
        }
    
    def _analyze_colors(self, img):
        """Extracts dominant colors using K-means"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape and sample pixels
            pixels = img.reshape(-1, 3)
            sample_size = min(5000, len(pixels))
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            sample_pixels = pixels[indices]
            
            # Find 5 dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(sample_pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            return {
                'palette': [self._rgb_to_hex(c) for c in colors],
                'diversity': self._calculate_color_diversity(colors)
            }
        except:
            return {
                'palette': ['#808080'],
                'diversity': 5
            }
    
    def _analyze_composition(self, img):
        """Analyzes visual composition using rule of thirds"""
        height, width = img.shape[:2]
        
        sections = []
        for i in range(3):
            for j in range(3):
                y1 = i * height // 3
                y2 = (i + 1) * height // 3
                x1 = j * width // 3
                x2 = (j + 1) * width // 3
                
                section = img[y1:y2, x1:x2]
                sections.append(np.mean(section))
        
        variance = np.std(sections)
        score = max(0, 100 - variance)
        
        return {
            'score': float(score),
            'balance': 'good' if score > 70 else 'needs_improvement'
        }
    
    def _analyze_text_readability(self, img):
        """Estimates text readability using edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        text_coverage = np.sum(edges > 0) / edges.size * 100
        
        return {
            'coverage': float(text_coverage),
            'estimated_size': 'adequate' if text_coverage > 5 else 'too_small'
        }
    
    def _calculate_image_score(self, brightness, contrast, colors, composition, text):
        """Calculates overall image quality score (0-100)"""
        score = 0
        
        # Brightness (20 points)
        if brightness['status'] == 'optimal':
            score += 20
        else:
            score += 10
        
        # Contrast (20 points)
        if contrast['status'] == 'high':
            score += 20
        elif contrast['status'] == 'medium':
            score += 15
        else:
            score += 8
        
        # Color diversity (15 points)
        score += min(15, colors['diversity'] * 3)
        
        # Composition (25 points)
        score += composition['score'] * 0.25
        
        # Text readability (20 points)
        if 5 < text['coverage'] < 30:
            score += 20
        else:
            score += 10
        
        return min(100, score)
    
    def _identify_issues(self, brightness, contrast, text):
        """Identifies specific issues with recommendations"""
        issues = []
        
        if brightness['status'] == 'too_dark':
            issues.append({
                'type': 'error',
                'category': 'Brightness',
                'issue': 'Image is too dark for optimal visibility',
                'impact': -15,
                'fix': 'Increase brightness by 20-30%',
                'technical': f'Current: {brightness["percentage"]:.1f}% → Target: 50-70%'
            })
        
        if brightness['status'] == 'too_bright':
            issues.append({
                'type': 'warning',
                'category': 'Brightness',
                'issue': 'Image is too bright, may appear washed out',
                'impact': -10,
                'fix': 'Reduce brightness by 15-25%',
                'technical': f'Current: {brightness["percentage"]:.1f}% → Target: 50-70%'
            })
        
        if contrast['status'] == 'low':
            issues.append({
                'type': 'warning',
                'category': 'Contrast',
                'issue': 'Low contrast reduces readability',
                'impact': -12,
                'fix': 'Increase contrast between text and background',
                'technical': f'Current ratio: {contrast["ratio"]:.2f} → Need: 4.5:1 minimum'
            })
        
        if text['estimated_size'] == 'too_small':
            issues.append({
                'type': 'error',
                'category': 'Text Size',
                'issue': 'Text appears too small for mobile viewing',
                'impact': -18,
                'fix': 'Increase font size by at least 25%',
                'technical': 'Recommended minimum: 16px for body text'
            })
        
        return issues
    
    def _rgb_to_hex(self, rgb):
        """Converts RGB to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def _calculate_color_diversity(self, colors):
        """Calculates color palette diversity"""
        distances = []
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                dist = np.linalg.norm(colors[i] - colors[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        return min(10, avg_distance / 50)