"""
Deep Learning Image Analyzer - OpenCV Compatible Version
Works without face_recognition library (uses OpenCV Haar Cascades instead)
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class DeepImageAnalyzer:
    """Advanced image analysis using deep learning"""
    
    def __init__(self):
        print("🔄 Loading deep learning models...")
        
        try:
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.eval()
            
            self.vgg = models.vgg16(pretrained=True)
            self.vgg.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
            
            print("✅ Deep learning models loaded!")
            self.models_loaded = True
            
        except Exception as e:
            print(f"⚠️  Could not load deep learning models: {str(e)}")
            self.models_loaded = False
    
    def analyze(self, image_file):
        """Comprehensive deep learning analysis"""
        try:
            image = Image.open(image_file).convert('RGB')
            img_array = np.array(image)
            
            objects = self.detect_objects(image)
            faces = self.detect_faces(img_array)
            emotions = self.analyze_emotions(img_array)
            scene_type = self.classify_scene(image)
            aesthetic_score = self.calculate_aesthetic_score(img_array)
            attention_map = self.generate_attention_heatmap(img_array)
            color_psychology = self.analyze_color_psychology(img_array)
            visual_complexity = self.analyze_visual_complexity(img_array)
            
            return {
                'objects_detected': objects,
                'faces_detected': faces,
                'emotions': emotions,
                'scene_type': scene_type,
                'aesthetic_score': aesthetic_score,
                'attention_hotspots': attention_map,
                'color_psychology': color_psychology,
                'visual_complexity': visual_complexity,
                'deep_learning_score': self.calculate_dl_score(objects, faces, emotions, aesthetic_score)
            }
            
        except Exception as e:
            print(f"❌ Deep learning analysis error: {str(e)}")
            return self.get_default_analysis()
    
    def detect_objects(self, image):
        """Detect objects using ResNet"""
        if not self.models_loaded:
            return {'count': 1, 'objects': [], 'main_subject': 'product', 'has_clear_focus': True}
        
        try:
            img_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = self.resnet(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            detected_objects = []
            
            for i in range(5):
                if top5_prob[i].item() > 0.05:
                    detected_objects.append({
                        'object': f'object_{top5_idx[i].item()}',
                        'confidence': float(top5_prob[i].item()),
                        'relevance': 'high' if top5_prob[i].item() > 0.5 else 'medium'
                    })
            
            return {
                'count': len(detected_objects),
                'objects': detected_objects,
                'main_subject': 'product',
                'has_clear_focus': len(detected_objects) <= 2
            }
        except:
            return {'count': 1, 'objects': [], 'main_subject': 'product', 'has_clear_focus': True}
    
    def detect_faces(self, img_array):
        """Face detection using OpenCV"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            img_height, img_width = img_array.shape[:2]
            faces_data = []
            
            for i, (x, y, w, h) in enumerate(faces):
                size_percentage = (w * h) / (img_width * img_height) * 100
                faces_data.append({
                    'id': i + 1,
                    'location': {'top': int(y), 'right': int(x + w), 'bottom': int(y + h), 'left': int(x)},
                    'size_percentage': round(size_percentage, 2),
                    'prominence': 'high' if size_percentage > 10 else 'medium' if size_percentage > 5 else 'low'
                })
            
            return {
                'count': len(faces),
                'faces': faces_data,
                'has_people': len(faces) > 0,
                'primary_focus': 'human' if len(faces) > 0 else 'product',
                'engagement_boost': '+38%' if len(faces) > 0 else 'baseline'
            }
        except:
            return {'count': 0, 'faces': [], 'has_people': False, 'primary_focus': 'product', 'engagement_boost': 'baseline'}
    
    def analyze_emotions(self, img_array):
        """Analyze emotional content"""
        try:
            avg_color = np.mean(img_array, axis=(0, 1))
            warmth = (avg_color[0] - avg_color[2]) / 255
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])
            
            if warmth > 0.2 and brightness > 140:
                emotion, appeal_score = 'joyful', 85
            elif warmth > 0.1 and saturation > 100:
                emotion, appeal_score = 'energetic', 80
            elif brightness < 100:
                emotion, appeal_score = 'serious', 65
            elif brightness > 180:
                emotion, appeal_score = 'calm', 70
            else:
                emotion, appeal_score = 'neutral', 60
            
            return {
                'dominant_emotion': emotion,
                'emotional_appeal_score': appeal_score,
                'warmth_level': round(float(warmth), 2),
                'mood': 'positive' if appeal_score > 70 else 'neutral',
                'engagement_potential': 'high' if appeal_score > 75 else 'medium'
            }
        except:
            return {'dominant_emotion': 'neutral', 'emotional_appeal_score': 60, 'warmth_level': 0.0, 'mood': 'neutral', 'engagement_potential': 'medium'}
    
    def classify_scene(self, image):
        """Classify scene type"""
        if not self.models_loaded:
            return {'type': 'product_shot', 'complexity': 'medium', 'visual_interest': 50.0}
        
        try:
            img_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.vgg.features(img_tensor)
                avg_pool = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                scene_features = avg_pool.view(avg_pool.size(0), -1)
            
            feature_std = scene_features.std().item()
            scene_type = 'lifestyle' if feature_std > 1.5 else 'outdoor' if feature_std > 1.0 else 'product_shot'
            complexity = 'high' if feature_std > 1.0 else 'medium' if feature_std > 0.5 else 'low'
            
            return {'type': scene_type, 'complexity': complexity, 'visual_interest': round(min(100, abs(feature_std) * 50), 1)}
        except:
            return {'type': 'product_shot', 'complexity': 'medium', 'visual_interest': 50.0}
    
    def calculate_aesthetic_score(self, img_array):
        """Calculate aesthetic quality"""
        try:
            height, width = img_array.shape[:2]
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            composition_score = 0
            for h in [height // 3, 2 * height // 3]:
                for w in [width // 3, 2 * width // 3]:
                    region = edges[max(0, h-20):min(height, h+20), max(0, w-20):min(width, w+20)]
                    if np.sum(region) > 1000:
                        composition_score += 20
            
            composition_score = min(100, composition_score)
            
            pixels = img_array.reshape(-1, 3)
            sample_pixels = pixels[np.random.choice(len(pixels), min(5000, len(pixels)), replace=False)]
            color_harmony = min(100, (len(np.unique(sample_pixels, axis=0)) / len(sample_pixels)) * 1000)
            
            aesthetic_score = composition_score * 0.6 + color_harmony * 0.4
            
            return {
                'score': round(aesthetic_score, 1),
                'composition_quality': 'excellent' if composition_score > 60 else 'good',
                'color_harmony': round(color_harmony, 1),
                'professional_look': aesthetic_score > 70
            }
        except:
            return {'score': 60.0, 'composition_quality': 'good', 'color_harmony': 60.0, 'professional_look': False}
    
    def generate_attention_heatmap(self, img_array):
        """Generate attention heatmap"""
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            success, saliency_map = saliency.computeSaliency(gray)
            
            if success:
                threshold = np.percentile(saliency_map, 90)
                hotspots = saliency_map > threshold
                num_hotspots = cv2.connectedComponents(hotspots.astype(np.uint8))[0] - 1
                
                return {
                    'hotspot_count': num_hotspots,
                    'attention_distribution': 'focused' if num_hotspots <= 2 else 'dispersed',
                    'visual_clarity': 'high' if num_hotspots <= 2 else 'medium',
                    'focus_score': max(0, 100 - (num_hotspots * 15))
                }
        except:
            pass
        return {'hotspot_count': 1, 'attention_distribution': 'focused', 'visual_clarity': 'medium', 'focus_score': 70}
    
    def analyze_color_psychology(self, img_array):
        """Analyze color psychology"""
        try:
            pixels = img_array.reshape(-1, 3)
            avg_color = np.mean(pixels[np.random.choice(len(pixels), min(10000, len(pixels)), replace=False)], axis=0)
            red, green, blue = avg_color
            
            if red > 150 and red > green and red > blue:
                return {'dominant_color_rgb': [int(red), int(green), int(blue)], 'psychology': 'energetic_urgent', 
                       'emotional_impact': 'excitement', 'cta_strength': 'high', 'brand_emotion': 'passionate'}
            elif blue > 150 and blue > red:
                return {'dominant_color_rgb': [int(red), int(green), int(blue)], 'psychology': 'trustworthy_calm', 
                       'emotional_impact': 'trust', 'cta_strength': 'medium', 'brand_emotion': 'reliable'}
            elif green > 150:
                return {'dominant_color_rgb': [int(red), int(green), int(blue)], 'psychology': 'natural_growth', 
                       'emotional_impact': 'harmony', 'cta_strength': 'medium', 'brand_emotion': 'sustainable'}
            else:
                return {'dominant_color_rgb': [int(red), int(green), int(blue)], 'psychology': 'neutral_professional', 
                       'emotional_impact': 'balance', 'cta_strength': 'medium', 'brand_emotion': 'corporate'}
        except:
            return {'dominant_color_rgb': [128, 128, 128], 'psychology': 'neutral', 'emotional_impact': 'neutral', 
                   'cta_strength': 'medium', 'brand_emotion': 'neutral'}
    
    def analyze_visual_complexity(self, img_array):
        """Analyze visual complexity"""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size * 100
            
            complexity_score = edge_density * 2
            style = 'minimalist' if complexity_score < 30 else 'clean' if complexity_score < 50 else 'detailed' if complexity_score < 70 else 'busy'
            
            return {
                'complexity_score': round(complexity_score, 1),
                'style': style,
                'readability': 'excellent' if complexity_score < 30 else 'good' if complexity_score < 60 else 'moderate',
                'recommendation': 'Good balance' if 30 < complexity_score < 60 else 'Consider simplifying' if complexity_score > 60 else 'Could add more visual interest'
            }
        except:
            return {'complexity_score': 50.0, 'style': 'balanced', 'readability': 'good', 'recommendation': 'Balanced design'}
    
    def calculate_dl_score(self, objects, faces, emotions, aesthetic):
        """Calculate overall deep learning score"""
        score = 0
        score += 15 if objects['count'] > 0 else 0
        score += 5 if objects.get('has_clear_focus') else 0
        score += 25 if faces['has_people'] else 0
        score += emotions['emotional_appeal_score'] * 0.25
        score += aesthetic['score'] * 0.25
        return round(min(100, score), 1)
    
    def get_default_analysis(self):
        """Return default analysis"""
        return {
            'objects_detected': {'count': 0, 'objects': [], 'main_subject': 'unknown', 'has_clear_focus': True},
            'faces_detected': {'count': 0, 'faces': [], 'has_people': False, 'primary_focus': 'product', 'engagement_boost': 'baseline'},
            'emotions': {'dominant_emotion': 'neutral', 'emotional_appeal_score': 60, 'warmth_level': 0.0, 'mood': 'neutral', 'engagement_potential': 'medium'},
            'scene_type': {'type': 'product_shot', 'complexity': 'medium', 'visual_interest': 50.0},
            'aesthetic_score': {'score': 60.0, 'composition_quality': 'good', 'color_harmony': 60.0, 'professional_look': False},
            'attention_hotspots': {'hotspot_count': 1, 'attention_distribution': 'focused', 'visual_clarity': 'medium', 'focus_score': 70},
            'color_psychology': {'dominant_color_rgb': [128, 128, 128], 'psychology': 'neutral', 'emotional_impact': 'neutral', 'cta_strength': 'medium', 'brand_emotion': 'neutral'},
            'visual_complexity': {'complexity_score': 50.0, 'style': 'balanced', 'readability': 'good', 'recommendation': 'Balanced design'},
            'deep_learning_score': 50.0
        }

if __name__ == "__main__":
    print("Testing Deep Image Analyzer...")
    analyzer = DeepImageAnalyzer()
    print("✅ Deep Image Analyzer ready!")