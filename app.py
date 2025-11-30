import os

# Production configuration
PORT = int(os.environ.get('PORT', 5001))
DEBUG = os.environ.get('FLASK_ENV') != 'production'

"""
CreativeIQ Backend Server - INTEGRATED VERSION with Dynamic IP
Complete AI-powered creative analysis system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
from get_ip import get_local_ip

# Import integrated analyzer
from integrated_analyzer import IntegratedAnalyzer

app = Flask(__name__)
CORS(app)

# Get dynamic IP
LOCAL_IP = get_local_ip()
PORT = 5001
BASE_URL = f"http://{LOCAL_IP}:{PORT}"

print(f"\n🌐 Dynamic IP detected: {LOCAL_IP}")
print(f"📍 Base URL: {BASE_URL}\n")

# Initialize the integrated AI system
print("🚀 Initializing CreativeIQ AI System...")
analyzer = IntegratedAnalyzer(enable_real_time_learning=True)
print("✅ System ready!\n")

# Store analysis results
analysis_history = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with system status"""
    system_status = analyzer.get_system_status()
    
    return jsonify({
        'status': 'healthy',
        'message': 'CreativeIQ AI System is running!',
        'version': '2.0.0',
        'capability_level': system_status['overall_capability'],
        'advanced_features': system_status['advanced_modules'],
        'server_info': {
            'ip': LOCAL_IP,
            'port': PORT,
            'base_url': BASE_URL
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_creative():
    """Main analysis endpoint with full AI integration"""
    try:
        image_file = request.files.get('image')
        caption = request.form.get('caption', '')
        
        print(f"\n{'='*60}")
        print(f"📥 NEW ANALYSIS REQUEST")
        print(f"{'='*60}")
        print(f"   - Image: {'✅ Yes' if image_file else '❌ No'}")
        print(f"   - Caption: {caption[:50]}..." if caption else "   - Caption: ❌ No")
        
        # Run comprehensive analysis
        results = analyzer.analyze_creative(image_file, caption)
        
        # Generate analysis ID
        analysis_id = f"analysis_{len(analysis_history) + 1}_{int(datetime.now().timestamp())}"
        
        # Store for future reference
        analysis_history[analysis_id] = {
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'has_feedback': False
        }
        
        # Prepare response
        response = {
            'success': True,
            'analysis_id': analysis_id,
            'data': results,
            'system_info': {
                'modules_used': results['metadata']['modules_used'],
                'prediction_method': results['metadata']['prediction_method'],
                'capability_level': analyzer.get_system_status()['overall_capability'],
                'server_url': BASE_URL
            }
        }
        
        print(f"✅ Analysis complete! ID: {analysis_id}")
        print(f"{'='*60}\n")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during analysis'
        }), 500

@app.route('/api/feedback', methods=['POST'])
def add_feedback():
    """Add actual performance data for continuous learning"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        actual_performance = data.get('actual_performance')
        
        if not analysis_id or not actual_performance:
            return jsonify({
                'success': False,
                'error': 'Missing analysis_id or actual_performance'
            }), 400
        
        if analysis_id not in analysis_history:
            return jsonify({
                'success': False,
                'error': 'Analysis ID not found'
            }), 404
        
        original_analysis = analysis_history[analysis_id]
        
        # Extract features for learning
        image_analysis = original_analysis['results']['basic_analysis'].get('image')
        text_analysis = original_analysis['results']['basic_analysis'].get('text')
        
        # Record for continuous learning
        analyzer.record_for_learning(
            image_analysis,
            text_analysis,
            actual_performance
        )
        
        # Mark as having feedback
        analysis_history[analysis_id]['has_feedback'] = True
        analysis_history[analysis_id]['actual_performance'] = actual_performance
        
        print(f"✅ Feedback recorded for {analysis_id}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded successfully',
            'learning_stats': analyzer.learner.get_learning_stats() if analyzer.learner else None
        }), 200
        
    except Exception as e:
        print(f"❌ Error recording feedback: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Platform statistics including AI system stats"""
    try:
        system_status = analyzer.get_system_status()
        
        total_analyses = len(analysis_history)
        with_feedback = sum(1 for a in analysis_history.values() if a.get('has_feedback', False))
        
        learning_stats = None
        if analyzer.learner:
            learning_stats = analyzer.learner.get_learning_stats()
        
        return jsonify({
            'total_analyses': total_analyses,
            'analyses_with_feedback': with_feedback,
            'system_capability': system_status['overall_capability'],
            'advanced_features_enabled': sum(system_status['advanced_modules'].values()),
            'learning_stats': learning_stats,
            'server_info': {
                'ip': LOCAL_IP,
                'base_url': BASE_URL
            },
            'top_improvements': [
                'Add Call-to-Action',
                'Increase Image Brightness',
                'Include Human Faces',
                'Add Urgency Signals',
                'Improve Brand Visibility'
            ]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status', methods=['GET'])
def system_status():
    """Detailed system status endpoint"""
    try:
        status = analyzer.get_system_status()
        status['server_info'] = {
            'ip': LOCAL_IP,
            'port': PORT,
            'base_url': BASE_URL
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning-stats', methods=['GET'])
def learning_stats():
    """Get real-time learning statistics"""
    try:
        if not analyzer.learner:
            return jsonify({
                'error': 'Real-time learning not enabled'
            }), 404
        
        stats = analyzer.learner.get_learning_stats()
        history = analyzer.learner.get_model_history()
        
        return jsonify({
            'stats': stats,
            'model_history': history,
            'server_info': {
                'ip': LOCAL_IP,
                'base_url': BASE_URL
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger-retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        if not analyzer.learner:
            return jsonify({
                'success': False,
                'error': 'Real-time learning not enabled'
            }), 404
        
        print("\n🔄 Manual retrain triggered via API")
        success = analyzer.learner.retrain_models()
        
        return jsonify({
            'success': success,
            'message': 'Retrain completed' if success else 'Retrain failed or not enough data',
            'learning_stats': analyzer.learner.get_learning_stats()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/server-info', methods=['GET'])
def server_info():
    """Get server connection information"""
    return jsonify({
        'ip': LOCAL_IP,
        'port': PORT,
        'base_url': BASE_URL,
        'endpoints': {
            'health': f'{BASE_URL}/api/health',
            'analyze': f'{BASE_URL}/api/analyze',
            'feedback': f'{BASE_URL}/api/feedback',
            'stats': f'{BASE_URL}/api/stats',
            'system_status': f'{BASE_URL}/api/system-status',
            'learning_stats': f'{BASE_URL}/api/learning-stats',
            'trigger_retrain': f'{BASE_URL}/api/trigger-retrain'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server',
        'server_info': {'ip': LOCAL_IP, 'base_url': BASE_URL}
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Something went wrong on our end',
        'server_info': {'ip': LOCAL_IP, 'base_url': BASE_URL}
    }), 500

if __name__ == '__main__':
    if DEBUG:
        from get_ip import get_local_ip
        LOCAL_IP = get_local_ip()
        BASE_URL = f"http://{LOCAL_IP}:{PORT}"
    else:
        LOCAL_IP = '0.0.0.0'
        BASE_URL = os.environ.get('BASE_URL', 'https://your-app.onrender.com')

    print("\n" + "="*60)
    print("🚀 Starting CreativeIQ AI System...")
    print("="*60)
    print(f"📍 Server IP: {LOCAL_IP}")
    print(f"📍 Server Port: {PORT}")
    print(f"🌐 Access URL: {BASE_URL}")
    print("="*60)
    print("\n⌨️  Press CTRL+C to stop the server\n")
    
    app.run(
        debug=DEBUG,
        host='0.0.0.0',
        port=PORT,
        threaded=True
    )
