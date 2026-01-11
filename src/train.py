"""
Model Training Module for Subway Delay Prediction

Trains HistGradientBoostingClassifier on engineered features and saves model artifacts.
"""

import joblib
import json
import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_engineering import feature_engineering


def train():
    """
    Train subway delay prediction model and save artifacts.
    
    Returns:
    --------
    dict : Training metrics and model information
    """
    print("=" * 60)
    print("SUBWAY DELAY PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and engineer features
    print("\n[1/5] Loading and engineering features...")
    subway_df, delay_rates = feature_engineering()
    
    # Define feature columns
    feature_columns = [
        # Basic temporal
        'DayOfWeek', 'Month', 'Hour', 'IsWeekend', 'IsRushHour',
        # Interaction
        'RushHour_Weekday', 'Weekend_Morning', 'Season',
        # Cyclical
        'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
        # Historical delay rates
        'Hour_DelayRate', 'Day_DelayRate', 'Station_DelayRate', 'Line_DelayRate', 'Code_DelayRate',
        # Time bins
        'Time_Night', 'Time_Morning', 'Time_Midday', 'Time_Evening', 'Time_Late'
    ]

    # Prepare feature matrix
    X = subway_df[feature_columns].copy()
    X['Line'] = subway_df['Line'].astype('category')
    X['Bound'] = subway_df['Bound'].astype('category')
    X['Station_Category'] = subway_df['Station_Category'].astype('category')

    # Tell the model which columns are categorical by their indices or names
    cat_features = ['Line', 'Bound', 'Station_Category']
    Y = subway_df['HasDelay']
    
    print(f"Feature matrix prepared!")
    print(f"  Shape: {X.shape}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Samples: {len(X):,}")
    print(f"\n  Class distribution:")
    print(f"    No Delay: {(Y == 0).sum():,} ({(Y == 0).sum()/len(Y)*100:.1f}%)")
    print(f"    Delay:    {(Y == 1).sum():,} ({(Y == 1).sum()/len(Y)*100:.1f}%)")

    # Split data
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2,
        random_state=42,
        stratify=Y
    )
    
    print(f"Data split completed!")
    print(f"  Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Testing set:    {len(X_test):,} samples ({len(X_test)/len(X)*100:.0f}%)")

    # Initialize model
    print("\n[3/5] Training HistGradientBoostingClassifier...")
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.01,
        max_iter=2500,
        max_leaf_nodes=127,
        max_depth=12,
        min_samples_leaf=100,
        l2_regularization=2.5,
        max_features=0.4,
        categorical_features=cat_features,
        early_stopping=True,
        n_iter_no_change=40,
        random_state=42,
        verbose=1
    )

    hgb.fit(X_train, Y_train)
    print("Model training complete!")

    # Evaluate model
    print("\n[4/5] Evaluating model performance...")
    y_pred = hgb.predict(X_test)
    y_pred_proba = hgb.predict_proba(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    cm = confusion_matrix(Y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred, target_names=['No Delay', 'Delay']))
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:,}")
    print(f"  False Positives: {cm[0,1]:,}")
    print(f"  False Negatives: {cm[1,0]:,}")
    print(f"  True Positives:  {cm[1,1]:,}")

    # Save model artifacts
    print(f"\n[5/5] Saving model artifacts...")
    
    # Create directories if they don't exist
    Path('models/trained').mkdir(parents=True, exist_ok=True)
    Path('models/metrics').mkdir(parents=True, exist_ok=True)
    
    # Save trained model
    joblib.dump(hgb, 'models/trained/trained_model.pkl')
    print("Saved trained model: models/trained/trained_model.pkl")
    
    # Save delay rates
    with open('models/delay_rates.json', 'w') as f:
        json.dump(delay_rates, f, indent=2)
    print("Saved delay rates: models/delay_rates.json")
    
    # Save model configuration
    model_config = {
        'algorithm': 'HistGradientBoostingClassifier',
        'training_date': datetime.datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'feature_columns': feature_columns,
        'categorical_features': cat_features,
        'total_features': X_train.shape[1],
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'hyperparameters': {
            'learning_rate': 0.01,
            'max_iter': 2500,
            'max_leaf_nodes': 127,
            'max_depth': 12,
            'min_samples_leaf': 100,
            'l2_regularization': 2.5,
            'max_features': 0.4
        }
    }
    
    with open('models/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    print("Saved model config: models/model_config.json")
    
    # Save detailed metrics
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'true_negatives': int(cm[0,0]),
            'false_positives': int(cm[0,1]),
            'false_negatives': int(cm[1,0]),
            'true_positives': int(cm[1,1])
        },
        'test_samples': len(X_test),
        'evaluation_date': datetime.datetime.now().isoformat()
    }
    
    with open('models/metrics/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved test metrics: models/metrics/test_metrics.json")
    
    print("Training complete!")

    
    return metrics


if __name__ == "__main__":
    train()

