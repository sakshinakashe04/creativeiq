"""
Brand Logo Recognition System
Detects and analyzes brand elements in images
"""

import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path

class BrandRecognizer:
    """Detect and analyze brand logos and visual identity"""
    
    def __init__(self):
        print("🔄 Initializing Brand Recognition System...")
        
        # Load known brand templates (in production, use a database)
        self.brand_database = self.load_brand_database()
        
        # Initialize feature detector
        try:
            self.feature_detector = cv2.SIFT_create()
        except:
            print("⚠️  SIFT not available, using ORB")
            self.feature_detector = cv2.ORB_create()
        
        print("✅ Brand Recognition System ready")
    
    def load_brand_database(self):
        """Load database of known brand logos"""
        return {
            'templates': [],
            'color_signatures': {
                'coca_cola': {'primary': (255, 0, 0), 'tolerance': 30},
                'pepsi': {'primary': (0, 82, 180), 'tolerance': 30},
                'mcdonalds': {'primary': (255, 188, 0), 'tolerance': 30},
                'starbucks': {'primary': (0, 98, 65), 'tolerance': 30},
                'nike': {'primary': (0, 0, 0), 'tolerance': 20},
                'apple': {'primary': (169, 169, 169), 'tolerance': 40},
                'google': {'primary': (66, 133, 244), 'tolerance': 30},
                'facebook': {'primary': (24, 119, 242), 'tolerance': 30},
                'amazon': {'primary': (255, 153, 0), 'tolerance': 30},
                'microsoft': {'primary': (0, 120, 212), 'tolerance': 30}
            }
        }
    
    def analyze(self, image_file):
        """
        Comprehensive brand analysis
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            Dictionary with brand analysis results
        """
        try:
            # Load image
            image = Image.open(image_file).convert('RGB')
            img_array = np.array(image)
            
            # Run all brand analyses
            logos = self.detect_logos(img_array)
            colors = self.analyze_brand_colors(img_array)
            text_branding = self.detect_brand_text(img_array)
            consistency = self.check_brand_consistency(img_array)
            visual_identity = self.analyze_visual_identity(img_array)
            placement = self.analyze_logo_placement(img_array, logos)
            
            return {
                'logos_detected': logos,
                'brand_colors': colors,
                'text_branding': text_branding,
                'brand_consistency': consistency,
                'visual_identity': visual_identity,
                'logo_placement': placement,
                'brand_strength_score': self.calculate_brand_score(
                    logos, colors, text_branding, consistency
                )
            }
            
        except Exception as e:
            print(f"❌ Brand recognition error: {str(e)}")
            return self.get_default_analysis()
    
    def detect_logos(self, img_array):
        """Detect logo presence and characteristics"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect logo-like regions using contour analysis
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logo_candidates = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Logo characteristics: reasonable size, not too extreme aspect ratio
                if 0.3 < aspect_ratio < 3.0 and 2000 < area < 100000:
                    # Check if region has distinct features
                    roi = img_array[y:y+h, x:x+w]
                    
                    # Check color uniqueness
                    avg_color = np.mean(roi, axis=(0, 1))
                    color_variance = np.var(roi, axis=(0, 1))
                    
                    # Calculate confidence based on multiple factors
                    size_score = min(1.0, area / 50000)
                    variance_score = min(1.0, np.mean(color_variance) / 1000)
                    confidence = (size_score * 0.6 + variance_score * 0.4)
                    
                    logo_candidates.append({
                        'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                        'area': int(area),
                        'aspect_ratio': round(aspect_ratio, 2),
                        'avg_color': [int(c) for c in avg_color],
                        'confidence': round(min(0.95, confidence), 2),
                        'position': self.get_logo_position(x, y, w, h, img_array.shape)
                    })
            
            # Sort by confidence
            logo_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'count': len(logo_candidates),
                'logos': logo_candidates[:5],  # Top 5
                'has_prominent_logo': len(logo_candidates) > 0 and logo_candidates[0]['confidence'] > 0.7,
                'primary_logo': logo_candidates[0] if logo_candidates else None
            }
            
        except Exception as e:
            print(f"Logo detection error: {str(e)}")
            return {'count': 0, 'logos': [], 'has_prominent_logo': False, 'primary_logo': None}
    
    def get_logo_position(self, x, y, w, h, img_shape):
        """Determine optimal logo placement"""
        img_h, img_w = img_shape[:2]
        
        # Calculate center
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        
        # Ideal positions: top-left, top-center, bottom-center
        positions = {
            'top_left': (0.15, 0.15),
            'top_center': (0.5, 0.15),
            'top_right': (0.85, 0.15),
            'center': (0.5, 0.5),
            'bottom_center': (0.5, 0.85),
            'bottom_left': (0.15, 0.85),
            'bottom_right': (0.85, 0.85)
        }
        
        # Find closest ideal position
        min_dist = float('inf')
        best_position = 'center'
        
        for pos_name, (ideal_x, ideal_y) in positions.items():
            dist = np.sqrt((center_x - ideal_x)**2 + (center_y - ideal_y)**2)
            if dist < min_dist:
                min_dist = dist
                best_position = pos_name
        
        placement_score = max(0, 100 - int(min_dist * 200))
        
        return {
            'actual': (round(center_x, 2), round(center_y, 2)),
            'ideal': best_position,
            'placement_score': placement_score,
            'is_optimal': placement_score > 70
        }
    
    def analyze_brand_colors(self, img_array):
        """Analyze brand color usage"""
        try:
            # Sample dominant colors
            pixels = img_array.reshape(-1, 3)
            sample_size = min(10000, len(pixels))
            sample_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
            
            # Get unique colors using K-means
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                kmeans.fit(sample_pixels)
                dominant_colors = kmeans.cluster_centers_.astype(int)
            except:
                # Fallback: simple color averaging
                dominant_colors = [np.mean(sample_pixels, axis=0).astype(int)]
            
            # Match against known brand colors
            brand_matches = []
            
            for brand, color_info in self.brand_database['color_signatures'].items():
                primary = np.array(color_info['primary'])
                tolerance = color_info['tolerance']
                
                for color in dominant_colors:
                    distance = np.linalg.norm(color - primary)
                    if distance < tolerance:
                        brand_matches.append({
                            'brand': brand,
                            'confidence': round(1 - (distance / tolerance), 2),
                            'color': [int(c) for c in color]
                        })
            
            # Analyze color consistency
            color_variance = np.var(sample_pixels, axis=0).mean()
            consistency_score = max(0, 100 - color_variance)
            
            return {
                'dominant_colors': [[int(c) for c in color] for color in dominant_colors],
                'brand_matches': sorted(brand_matches, key=lambda x: x['confidence'], reverse=True)[:3],
                'color_consistency': round(consistency_score, 1),
                'professional_palette': consistency_score > 60,
                'has_signature_color': len(brand_matches) > 0,
                'color_count': len(dominant_colors)
            }
            
        except Exception as e:
            print(f"Color analysis error: {str(e)}")
            return {
                'dominant_colors': [],
                'brand_matches': [],
                'color_consistency': 50.0,
                'professional_palette': False,
                'has_signature_color': False,
                'color_count': 0
            }
    
    def detect_brand_text(self, img_array):
        """Detect text-based branding elements"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect text regions using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find text-like contours
            edges = cv2.Canny(morph, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Text-like: horizontal, reasonable size
                if aspect_ratio > 2 and 1000 < area < 50000:
                    text_regions.append({
                        'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                        'type': 'horizontal_text',
                        'size': 'large' if area > 20000 else 'medium' if area > 5000 else 'small'
                    })
            
            # Determine text prominence
            total_text_area = sum(r['bbox']['w'] * r['bbox']['h'] for r in text_regions)
            img_area = img_array.shape[0] * img_array.shape[1]
            text_coverage = (total_text_area / img_area * 100) if img_area > 0 else 0
            
            return {
                'text_region_count': len(text_regions),
                'text_regions': text_regions[:5],  # Top 5
                'text_coverage_percent': round(text_coverage, 2),
                'text_prominence': 'high' if text_coverage > 15 else 'medium' if text_coverage > 5 else 'low',
                'brand_name_visible': len(text_regions) > 0,
                'readable': text_coverage > 5 and text_coverage < 30
            }
            
        except Exception as e:
            print(f"Text branding error: {str(e)}")
            return {
                'text_region_count': 0,
                'text_regions': [],
                'text_coverage_percent': 0,
                'text_prominence': 'low',
                'brand_name_visible': False,
                'readable': False
            }
    
    def check_brand_consistency(self, img_array):
        """Check overall brand consistency"""
        try:
            # Check color consistency
            std_per_channel = np.std(img_array, axis=(0, 1))
            color_consistency = 100 - np.mean(std_per_channel) / 2.55
            
            # Check layout balance
            h, w = img_array.shape[:2]
            left_half = img_array[:, :w//2]
            right_half = img_array[:, w//2:]
            
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            
            balance_score = 100 - abs(left_mean - right_mean)
            
            # Check top vs bottom balance
            top_half = img_array[:h//2, :]
            bottom_half = img_array[h//2:, :]
            
            top_mean = np.mean(top_half)
            bottom_mean = np.mean(bottom_half)
            
            vertical_balance = 100 - abs(top_mean - bottom_mean)
            
            # Overall consistency
            consistency_score = (color_consistency * 0.5 + balance_score * 0.3 + vertical_balance * 0.2)
            
            return {
                'overall_consistency': round(consistency_score, 1),
                'color_consistency': round(color_consistency, 1),
                'horizontal_balance': round(balance_score, 1),
                'vertical_balance': round(vertical_balance, 1),
                'professional_appearance': consistency_score > 70,
                'rating': 'excellent' if consistency_score > 80 else 'good' if consistency_score > 60 else 'needs_improvement'
            }
            
        except Exception as e:
            print(f"Consistency check error: {str(e)}")
            return {
                'overall_consistency': 50.0,
                'color_consistency': 50.0,
                'horizontal_balance': 50.0,
                'vertical_balance': 50.0,
                'professional_appearance': False,
                'rating': 'unknown'
            }
    
    def analyze_visual_identity(self, img_array):
        """Analyze overall visual brand identity"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Edge density (clean vs busy)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size * 100
            
            # Color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])
            
            # Brightness
            brightness = np.mean(gray)
            
            # Determine style
            if edge_density < 5 and saturation < 100:
                style = 'minimalist'
                modern_score = 95
            elif saturation > 150:
                style = 'vibrant'
                modern_score = 85
            elif edge_density > 15:
                style = 'detailed'
                modern_score = 70
            else:
                style = 'balanced'
                modern_score = 80
            
            return {
                'style': style,
                'modern_score': modern_score,
                'edge_density': round(edge_density, 2),
                'saturation_level': round(saturation, 2),
                'brightness_level': round(brightness, 2),
                'brand_personality': self.infer_brand_personality(style, saturation),
                'target_demographic': self.infer_demographic(style, saturation),
                'design_trend': self.identify_design_trend(edge_density, saturation)
            }
            
        except Exception as e:
            print(f"Visual identity error: {str(e)}")
            return {
                'style': 'unknown',
                'modern_score': 50,
                'edge_density': 0,
                'saturation_level': 0,
                'brightness_level': 0,
                'brand_personality': 'neutral',
                'target_demographic': 'general',
                'design_trend': 'classic'
            }
    
    def analyze_logo_placement(self, img_array, logos):
        """Analyze logo placement effectiveness"""
        if not logos['has_prominent_logo']:
            return {
                'has_logo': False,
                'placement_quality': 'none',
                'visibility_score': 0,
                'recommendations': ['Add a prominent logo for brand recognition']
            }
        
        primary = logos['primary_logo']
        position = primary['position']
        
        recommendations = []
        
        # Check placement score
        if position['placement_score'] < 50:
            recommendations.append(f"Consider moving logo to {position['ideal']} position")
        
        # Check size
        img_area = img_array.shape[0] * img_array.shape[1]
        logo_percentage = (primary['area'] / img_area) * 100
        
        if logo_percentage < 1:
            recommendations.append("Logo is too small, increase size by 50-100%")
        elif logo_percentage > 15:
            recommendations.append("Logo is too large, reduce size for better balance")
        
        visibility_score = min(100, position['placement_score'] + (logo_percentage * 10))
        
        return {
            'has_logo': True,
            'placement_quality': 'excellent' if visibility_score > 80 else 'good' if visibility_score > 60 else 'poor',
            'visibility_score': round(visibility_score, 1),
            'logo_size_percent': round(logo_percentage, 2),
            'optimal_size': 2 < logo_percentage < 10,
            'recommendations': recommendations if recommendations else ['Logo placement is optimal']
        }
    
    def infer_brand_personality(self, style, saturation):
        """Infer brand personality from visual characteristics"""
        if style == 'minimalist':
            return 'sophisticated_modern'
        elif style == 'vibrant' and saturation > 150:
            return 'energetic_youthful'
        elif style == 'detailed':
            return 'traditional_trustworthy'
        else:
            return 'balanced_professional'
    
    def infer_demographic(self, style, saturation):
        """Infer target demographic"""
        if style == 'minimalist':
            return 'millennials_gen_z'
        elif saturation > 150:
            return 'youth_teens'
        elif style == 'detailed':
            return 'mature_professionals'
        else:
            return 'broad_appeal'
    
    def identify_design_trend(self, edge_density, saturation):
        """Identify current design trend"""
        if edge_density < 5:
            return 'flat_design'
        elif saturation > 150:
            return 'material_design'
        elif edge_density > 15:
            return 'skeuomorphic'
        else:
            return 'neo_brutalism'
    
    def calculate_brand_score(self, logos, colors, text, consistency):
        """Calculate overall brand strength score"""
        score = 0
        
        # Logo presence and quality (30 points)
        if logos['has_prominent_logo']:
            score += 25
            if logos['primary_logo']['confidence'] > 0.8:
                score += 5
        elif logos['count'] > 0:
            score += 15
        
        # Color consistency and recognition (25 points)
        score += colors['color_consistency'] * 0.20
        if colors['has_signature_color']:
            score += 5
        
        # Text branding (20 points)
        if text['brand_name_visible']:
            score += 15
            if text['readable']:
                score += 5
        elif text['text_region_count'] > 0:
            score += 10
        
        # Overall consistency (25 points)
        score += consistency['overall_consistency'] * 0.25
        
        return round(min(100, score), 1)
    
    def get_default_analysis(self):
        """Return default analysis on error"""
        return {
            'logos_detected': {'count': 0, 'logos': [], 'has_prominent_logo': False, 'primary_logo': None},
            'brand_colors': {'dominant_colors': [], 'brand_matches': [], 'color_consistency': 50.0, 
                           'professional_palette': False, 'has_signature_color': False, 'color_count': 0},
            'text_branding': {'text_region_count': 0, 'text_regions': [], 'text_coverage_percent': 0, 
                            'text_prominence': 'low', 'brand_name_visible': False, 'readable': False},
            'brand_consistency': {'overall_consistency': 50.0, 'color_consistency': 50.0, 
                                'horizontal_balance': 50.0, 'vertical_balance': 50.0, 
                                'professional_appearance': False, 'rating': 'unknown'},
            'visual_identity': {'style': 'unknown', 'modern_score': 50, 'edge_density': 0, 
                              'saturation_level': 0, 'brightness_level': 0, 
                              'brand_personality': 'neutral', 'target_demographic': 'general', 'design_trend': 'classic'},
            'logo_placement': {'has_logo': False, 'placement_quality': 'none', 'visibility_score': 0, 
                             'recommendations': []},
            'brand_strength_score': 50.0
        }

# Test
if __name__ == "__main__":
    recognizer = BrandRecognizer()
    print("✅ Brand Recognition System initialized successfully!")
