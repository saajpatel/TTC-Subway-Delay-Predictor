"""
REST API for Subway Delay Prediction

Provides HTTP endpoints for making delay predictions.

Usage:
    python api.py

Endpoints:
    POST /predict - Single prediction
    POST /predict_day - 24-hour prediction
    GET /health - Health check
"""

from flask import Flask, request, jsonify
from src.predict import SubwayDelayPredictor
import traceback

app = Flask(__name__)

# Initialize predictor once at startup
print("Initializing predictor...")
try:
    predictor = SubwayDelayPredictor()
    print("Predictor ready!")
except Exception as e:
    print(f"Failed to load predictor: {e}")
    predictor = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if predictor else 'unhealthy',
        'service': 'Subway Delay Prediction API',
        'model_loaded': predictor is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a single delay prediction
    
    Request body:
    {
        "Date": "2026-01-15",
        "Time": "08:30",
        "Station": "BLOOR YONGE STATION",
        "Line": "BD",
        "Code": "MUSC",
        "Bound": "W"
    }
    
    Returns:
    {
        "prediction": "Delay" or "No Delay",
        "delay_probability": 0.75,
        "confidence": {...}
    }
    """
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Date', 'Time', 'Station', 'Line', 'Code', 'Bound']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields
            }), 400
        
        # Make prediction
        result = predictor.predict(data)
        
        return jsonify({
            'success': True,
            'input': data,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/predict_day', methods=['POST'])
def predict_day():
    """
    Predict delays for all 24 hours of a given day
    
    Request body:
    {
        "Date": "2026-01-15",
        "Station": "BLOOR YONGE STATION",
        "Line": "BD",
        "Code": "MUSC",
        "Bound": "W"
    }
    
    Returns:
    {
        "hourly_predictions": [
            {"hour": 0, "delay_probability": 0.45, ...},
            {"hour": 1, "delay_probability": 0.42, ...},
            ...
        ]
    }
    """
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Date', 'Station', 'Line', 'Code', 'Bound']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields
            }), 400
        
        # Make prediction
        result = predictor.predict_day_hourly(
            target_date=data['Date'],
            station=data['Station'],
            line=data['Line'],
            code=data['Code'],
            bound=data['Bound']
        )
        
        return jsonify({
            'success': True,
            'input': data,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Make predictions for multiple inputs at once
    
    Request body:
    {
        "predictions": [
            {"Date": "2026-01-15", "Time": "08:30", ...},
            {"Date": "2026-01-15", "Time": "17:00", ...},
            ...
        ]
    }
    """
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        predictions_input = data.get('predictions', [])
        
        if not predictions_input:
            return jsonify({'error': 'No predictions provided'}), 400
        
        results = []
        for idx, pred_input in enumerate(predictions_input):
            try:
                result = predictor.predict(pred_input)
                results.append({
                    'index': idx,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(predictions_input),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚇 SUBWAY DELAY PREDICTION API")
    print("="*60)
    print("\nEndpoints:")
    print("GET  /health         - Health check")
    print("POST /predict        - Single prediction")
    print("POST /predict_day    - 24-hour prediction")
    print("POST /predict_batch  - Batch predictions")
    print("\nStarting server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
