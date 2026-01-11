"""
Prediction Module for Subway Delay Prediction

Load the trained model and make predictions on new data.
"""

import pandas as pd
import numpy as np
import json
import joblib


class SubwayDelayPredictor:
    """
    Subway delay prediction using trained HistGradientBoostingClassifier.
    """
    
    def __init__(self, model_path='models/trained/trained_model.pkl',
                 delay_rates_path='models/delay_rates.json',
                 config_path='models/model_config.json'):
        """
        Initialize predictor by loading trained model and artifacts.
        
        Parameters:
        -----------
        model_path : str
            Path to trained model pickle file
        delay_rates_path : str
            Path to delay rates JSON file
        config_path : str
            Path to model configuration JSON file
        """
        print("Loading model artifacts...")
        
        # Load trained model
        self.model = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
        
        # Load delay rates
        with open(delay_rates_path, 'r') as f:
            self.delay_rates = json.load(f)
        print(f"Loaded delay rates: {delay_rates_path}")
        
        # Load model config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print(f"Loaded config: {config_path}")
        
        print(f"\nModel Info:")
        print(f"  Algorithm: {self.config['algorithm']}")
        print(f"  Trained: {self.config['training_date']}")
        print(f"  Accuracy: {self.config['accuracy']:.2%}")
        print(f"  Features: {self.config['total_features']}")
    
    def _engineer_features(self, input_data):
        """
        Engineer features from raw input data.
        
        Parameters:
        -----------
        input_data : dict
            Raw input with keys: Date, Time, Station, Line, Code, Bound
        
        Returns:
        --------
        pd.DataFrame : Feature matrix ready for prediction
        """
        # Parse datetime
        date = pd.to_datetime(input_data['Date'])
        time = pd.to_datetime(input_data['Time'], format='%H:%M')
        
        # Basic temporal features
        day_of_week = date.dayofweek
        month = date.month
        hour = time.hour
        is_weekend = int(day_of_week >= 5)
        is_rush_hour = int(hour in [7, 8, 9, 17, 18])
        
        # Interaction features
        rush_hour_weekday = is_rush_hour * (1 - is_weekend)
        weekend_morning = is_weekend * int(hour < 12)
        season = 0 if month in [12, 1, 2] else 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8] else 3
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Historical delay rates
        hour_delay_rate = self.delay_rates['hour'].get(str(hour), 0.5)
        day_delay_rate = self.delay_rates['day'].get(str(day_of_week), 0.5)
        station_delay_rate = self.delay_rates['station'].get(input_data['Station'], 0.5)
        line_delay_rate = self.delay_rates['line'].get(input_data['Line'], 0.5)
        code_delay_rate = self.delay_rates['code'].get(input_data['Code'], 0.5)
        
        # Time bins
        if hour <= 6:
            time_bin = 'Night'
        elif hour <= 10:
            time_bin = 'Morning'
        elif hour <= 16:
            time_bin = 'Midday'
        elif hour <= 19:
            time_bin = 'Evening'
        else:
            time_bin = 'Late'
        
        # Create feature dictionary
        features = {
            'DayOfWeek': day_of_week,
            'Month': month,
            'Hour': hour,
            'IsWeekend': is_weekend,
            'IsRushHour': is_rush_hour,
            'RushHour_Weekday': rush_hour_weekday,
            'Weekend_Morning': weekend_morning,
            'Season': season,
            'Hour_sin': hour_sin,
            'Hour_cos': hour_cos,
            'DayOfWeek_sin': day_sin,
            'DayOfWeek_cos': day_cos,
            'Hour_DelayRate': hour_delay_rate,
            'Day_DelayRate': day_delay_rate,
            'Station_DelayRate': station_delay_rate,
            'Line_DelayRate': line_delay_rate,
            'Code_DelayRate': code_delay_rate,
            'Time_Night': int(time_bin == 'Night'),
            'Time_Morning': int(time_bin == 'Morning'),
            'Time_Midday': int(time_bin == 'Midday'),
            'Time_Evening': int(time_bin == 'Evening'),
            'Time_Late': int(time_bin == 'Late'),
            'Line': input_data['Line'],
            'Bound': input_data['Bound'],
            'Station_Category': input_data.get('Station_Category', 'Other')
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Set categorical types
        df['Line'] = df['Line'].astype('category')
        df['Bound'] = df['Bound'].astype('category')
        df['Station_Category'] = df['Station_Category'].astype('category')
        
        return df
    
    def predict(self, input_data):
        """
        Make a single delay prediction.
        
        Parameters:
        -----------
        input_data : dict
            Input with keys: Date, Time, Station, Line, Code, Bound
            Example: {
                'Date': '2027-01-15',
                'Time': '08:30',
                'Station': 'BLOOR YONGE STATION',
                'Line': 'BD',
                'Code': 'MUSC',
                'Bound': 'W'
            }
        
        Returns:
        --------
        dict : Prediction result with delay probability
        """
        # Engineer features
        X = self._engineer_features(input_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            'prediction': 'Delay' if prediction == 1 else 'No Delay',
            'delay_probability': float(probabilities[1]),
            'no_delay_probability': float(probabilities[0]),
            'confidence': {
                'no_delay': float(probabilities[0]),
                'delay': float(probabilities[1])
            }
        }
    
    def predict_day_hourly(self, target_date, station='BLOOR YONGE STATION', 
                          line='BD', code='MUSC', bound='W'):
        """
        Predict delays for all 24 hours of a given date.
        
        Parameters:
        -----------
        target_date : str or datetime
            The date to predict (e.g., '2027-01-23')
        station : str
            Station to use
        line : str
            Line to use
        code : str
            Incident code to use
        bound : str
            Direction bound
        
        Returns:
        --------
        list of dict : 24 predictions (one per hour)
        """
        predictions = []
        
        for hour in range(24):
            input_data = {
                'Date': target_date,
                'Time': f'{hour:02d}:00',
                'Station': station,
                'Line': line,
                'Code': code,
                'Bound': bound
            }
            result = self.predict(input_data)
            predictions.append({
                'hour': hour,
                'delay_probability': result['confidence']['delay'] * 100,
                'prediction': result['prediction'],
                'time': f'{hour:02d}:00'
            })
        
        return predictions
    
    def predict_batch(self, input_list):
        """
        Make predictions for multiple inputs.
        
        Parameters:
        -----------
        input_list : list of dict
            List of input dictionaries
        
        Returns:
        --------
        list of dict : Predictions for each input
        """
        return [self.predict(input_data) for input_data in input_list]


def demo_predictions():
    """
    Demo function showing how to use the predictor.
    """
    print("=" * 60)
    print("SUBWAY DELAY PREDICTION DEMO")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SubwayDelayPredictor()
    
    print("\n" + "=" * 60)
    print("Example 1: Single Prediction")
    print("=" * 60)
    
    # Example prediction
    input_data = {
        'Date': '2027-01-15',
        'Time': '08:30',
        'Station': 'BLOOR YONGE STATION',
        'Line': 'BD',
        'Code': 'MUSC',
        'Bound': 'W'
    }
    
    print(f"\nInput: {input_data}")
    result = predictor.predict(input_data)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Delay Probability: {result['delay_probability']:.1%}")
    print(f"No Delay Probability: {result['no_delay_probability']:.1%}")
    
    print("\n" + "=" * 60)
    print("Example 2: 24-Hour Prediction")
    print("=" * 60)
    
    hourly_predictions = predictor.predict_day_hourly('2027-01-15', line='BD')
    
    print(f"\nDelay probabilities for January 15, 2027:\n")
    print(f"{'Hour':<10} {'Time':<10} {'Delay %':<12} {'Prediction'}")
    print("-" * 50)
    
    for pred in hourly_predictions:
        print(f"{pred['hour']:<10} {pred['time']:<10} {pred['delay_probability']:>6.1f}%     {pred['prediction']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_predictions()
