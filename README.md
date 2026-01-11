# 🚇 Toronto Subway Delay Predictor

A machine learning system that predicts TTC (Toronto Transit Commission) subway delays using historical data from 2018-2025. Built with scikit-learn and deployed with Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](placeholder)

## 🎯 Features

- **Real-time Predictions**: Predict delays for specific dates, times, and stations
- **24-Hour Forecasts**: Visualize delay probabilities throughout the day
- **Interactive Dashboard**: User-friendly web interface with charts and metrics
- **REST API**: Programmatic access to predictions
- **High Accuracy**: 83.4% accuracy on test data

## 🌐 Live Demo

**Try it now**: [temp placeholder]

No installation required - use the web app directly!

## 📊 Model Performance

- **Algorithm**: HistGradientBoostingClassifier
- **Accuracy**: 83.42%
- **Training Data**: 96,873 delay records (2018-2025)
- **Features**: 25 engineered features including:
  - Temporal patterns (hour, day, month, season)
  - Cyclical encoding (for time continuity)
  - Historical delay rates by station/line/hour
  - Rush hour and weekend interactions

## 🚀 Quick Start

### Option 1: Use the Hosted App (Recommended)

Just visit: **[placeholder]**

### Option 2: Run Locally

#### Prerequisites

- Python 3.8+
- pip

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Option 3: Run the REST API

```bash
python api.py
```

The API will be available at `http://localhost:5000`

## 📖 Usage

### Web Dashboard

1. Navigate to the **Single Prediction** tab
2. Select date, time, station, line, and other parameters
3. Click "Predict Delay" to get the probability
4. Or use the **24-Hour Forecast** tab to see delay patterns throughout the day

### REST API

**Single Prediction**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "2026-01-15",
    "Time": "08:30",
    "Station": "BLOOR YONGE STATION",
    "Line": "BD",
    "Code": "MUSC",
    "Bound": "W"
  }'
```

**24-Hour Forecast**
```bash
curl -X POST http://localhost:5000/predict_day \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "2026-01-15",
    "Station": "BLOOR YONGE STATION",
    "Line": "BD",
    "Code": "MUSC",
    "Bound": "W"
  }'
```

**Health Check**
```bash
curl http://localhost:5000/health
```

### Python Code

```python
from src.predict import SubwayDelayPredictor

# Initialize predictor
predictor = SubwayDelayPredictor()

# Make a prediction
result = predictor.predict({
    'Date': '2026-01-15',
    'Time': '08:30',
    'Station': 'BLOOR YONGE STATION',
    'Line': 'YU',
    'Code': 'SUO',
    'Bound': 'N'
})

print(f"Prediction: {result['prediction']}")
print(f"Delay Probability: {result['delay_probability']:.1%}")
```

## 🏗️ Project Structure

```
ml/
├── app.py                          # Streamlit dashboard
├── api.py                          # REST API (Flask)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/
│   ├── data_cleaning.py           # Data preprocessing
│   ├── feature_engineering.py     # Feature creation
│   ├── train.py                   # Model training
│   └── predict.py                 # Prediction logic
├── models/
│   ├── trained/
│   │   └── trained_model.pkl      # Trained model
│   ├── delay_rates.json           # Historical delay rates
│   ├── model_config.json          # Model metadata
│   └── metrics/
│       └── test_metrics.json      # Performance metrics
├── data/
│   ├── raw/                       # Original CSV/XLSX files
│   ├── processed/                 # Cleaned data by year
│   └── final/
│       └── final.csv              # Merged dataset
└── notebooks/
    └── model_development.ipynb    # Exploratory analysis
```

## 🔧 Development

### Train a New Model

If you have new data or want to retrain:

1. **Add data** to `data/raw/` (CSV or XLSX format)

2. **Clean and merge data**
   ```bash
   python src/data_cleaning.py
   ```

3. **Train the model**
   ```bash
   python src/train.py
   ```

This will:
- Engineer features
- Train the model
- Save model artifacts to `models/`
- Display performance metrics

### Model Features

The model uses 25 engineered features:

**Temporal Features:**
- Hour, Day of Week, Month, Year
- Weekend indicator, Rush hour indicator
- Season (Winter, Spring, Summer, Fall)

**Cyclical Encoding:**
- Hour sin/cos (24-hour cycle)
- Day of week sin/cos (7-day cycle)

**Historical Delay Rates:**
- By hour, day, station, line, and incident code

**Interaction Features:**
- Rush hour × Weekday
- Weekend × Morning
- Time bins (Night, Morning, Midday, Evening, Late)

**Categorical Features:**
- Line (YU, BD, SHP, SRT)
- Direction (N, E, S, W)
- Station category (top 10 stations + Other)

## 📊 Dataset

- **Source**: TTC Subway Delay Data (2018-2025)
- **Size**: 96,873 records
- **Time Period**: 8 years
- **Features**: Station, Line, Time, Date, Delay Duration, Incident Code

## 🛠️ Technology Stack

- **Machine Learning**: scikit-learn (HistGradientBoostingClassifier)
- **Web Framework**: Streamlit (dashboard), Flask (API)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: Streamlit Community Cloud

## 📈 Model Details

### Hyperparameters

- Learning rate: 0.01
- Max iterations: 2,500
- Max depth: 12
- Max leaf nodes: 127
- Min samples per leaf: 100
- L2 regularization: 2.5
- Max features: 0.4

### Performance

- **Accuracy**: 83.42%
- **Training samples**: 77,498
- **Test samples**: 19,375
- **Train/Test split**: 80/20 with stratification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request by forking the repository.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Toronto Transit Commission (TTC) for providing public delay data!

## 📧 Contact

For questions or feedback, please open an issue on GitHub.