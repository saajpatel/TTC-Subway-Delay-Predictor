# Transit Dashboard Project - Getting Started

## 🚀 Quick Start Guide

### Phase 1: Environment Setup (15 min)

1. **Create virtual environment:**
```bash
cd ml
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Phase 2: Data Collection (1-2 hours)

1. **Download TTC data:**
   - Visit: https://open.toronto.ca/dataset/ttc-subway-delay-data/
   - Download CSV files for 2023-2024 (or available years)
   - Save to `data/` folder as `ttc_delays.csv`

2. **Verify data:**
```bash
python src/data_collection.py
```

### Phase 3: Data Cleaning & Processing (2-3 hours)

1. **Open Jupyter notebook for exploration:**
```bash
jupyter notebook
```

2. **Clean the data:**
```python
import pandas as pd
from src.data_cleaning import clean_delay_data, feature_engineering

# Load raw data
df = pd.read_csv('data/ttc_delays.csv')

# Clean
df_clean = clean_delay_data(df)

# Add features
df_final = feature_engineering(df_clean)

# Save
df_final.to_csv('data/processed_data.csv', index=False)
```

### Phase 4: Database Setup (1 hour)

```python
from src.database import TransitDatabase

# Create database
db = TransitDatabase()

# Load processed data
df = pd.read_csv('data/processed_data.csv')
db.load_data_to_db(df)

# Test queries
stats = db.get_delay_stats_by_route()
print(stats)
```

### Phase 5: Analysis (2-3 hours)

```python
from src.analysis import analyze_delay_trends, peak_hour_analysis, generate_insights

df = pd.read_csv('data/processed_data.csv')

# Run analyses
trends = analyze_delay_trends(df)
peak_analysis = peak_hour_analysis(df)
insights = generate_insights(df)

print(insights)
```

### Phase 6: Build Dashboard (3-4 hours)

1. **Run the dashboard:**
```bash
streamlit run app.py
```

2. **Customize visualizations in app.py**

3. **Deploy to Streamlit Cloud (free):**
   - Push to GitHub
   - Connect at share.streamlit.io
   - Done!

---

## 📊 Expected Columns in TTC Data

TTC delay data typically includes:
- `Date` - Date of delay
- `Time` - Time of delay
- `Day` - Day of week
- `Station` - Station where delay occurred
- `Code` - Delay code/reason
- `Min Delay` - Minutes of delay
- `Min Gap` - Minutes gap between trains
- `Bound` - Direction (N/S/E/W)
- `Line` - Subway line (1/2/3/4)

You'll need to adapt cleaning scripts based on actual column names.

---

## 🎯 Resume-Worthy Features to Add

Once basics work, add these for resume impact:

1. **SQL Optimization:**
   - Create indexes on frequently queried columns
   - Write complex JOIN queries
   - Add query performance metrics

2. **Advanced Visualizations:**
   - Animated time-series plots
   - Interactive Plotly charts
   - Geospatial heatmaps with folium

3. **Statistical Analysis:**
   - Correlation with weather data
   - Predictive modeling (which routes will delay?)
   - Time series forecasting

4. **Dashboard Features:**
   - Real-time filters
   - Export to PDF reports
   - Email alerts for delay patterns

---

## 💡 Tips

- **Start simple:** Get basic pipeline working first
- **Document as you go:** Comment your SQL queries
- **Track metrics:** Note processing time, data size, accuracy
- **Deploy early:** Get it online for real usage metrics
- **Share it:** Post on LinkedIn/Reddit to get actual users

---

## 🐛 Troubleshooting

**"ModuleNotFoundError":**
```bash
pip install -r requirements.txt
```

**"Data not loading":**
- Check file path in data_collection.py
- Verify CSV column names match your cleaning script

**"Streamlit not working":**
```bash
pip install --upgrade streamlit
streamlit run app.py --logger.level=debug
```

---

## Next Steps

1. ✅ Download TTC data
2. ✅ Run data cleaning
3. ✅ Load into database
4. ✅ Create visualizations
5. ✅ Build dashboard
6. ✅ Deploy online
7. ✅ Write resume bullets!
