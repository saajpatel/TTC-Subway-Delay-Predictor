"""
Streamlit Dashboard for Subway Delay Prediction
Interactive web interface for predicting TTC delays
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, time
from src.predict import SubwayDelayPredictor

# Page config
st.set_page_config(
    page_title="Subway Delay Predictor",
    page_icon="🚇",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .delay-yes {
        color: #d62728;
    }
    .delay-no {
        color: #2ca02c;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    return SubwayDelayPredictor()

# Main app
def main():
    st.title("🚇 Toronto Subway Delay Predictor")
    st.markdown("Predict TTC subway delays using machine learning")
    
    try:
        predictor = load_predictor()
        
        # Show model info in sidebar
        st.sidebar.header("📊 Model Information")
        st.sidebar.metric("Algorithm", predictor.config['algorithm'])
        st.sidebar.metric("Accuracy", f"{predictor.config['accuracy']:.2%}")
        st.sidebar.metric("Features", predictor.config['total_features'])
        st.sidebar.caption(f"Trained: {predictor.config['training_date'][:10]}")
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📅 24-Hour Forecast"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.header("Make a Single Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_date = st.date_input(
                "Date",
                value=date.today(),
                min_value=date(2018, 1, 1),
                max_value=date(2030, 12, 31)
            )
            
            current_time = datetime.now()
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                hour = st.number_input("Hour", min_value=0, max_value=23, value=current_time.hour, step=1)
            with time_col2:
                minute = st.number_input("Minute", min_value=0, max_value=59, value=current_time.minute, step=1)
            
            input_time = time(hour, minute)
            
            station = st.text_input(
                "Station",
                value="BLOOR YONGE STATION",
                help="Enter the station name (e.g., BLOOR YONGE STATION)"
            )
        
        with col2:
            line = st.selectbox(
                "Line",
                ["YU", "BU", "SHP", "SRT"],
                help="YU=Yonge-University, BD=Bloor-Danforth, SHP=Sheppard, SRT=Scarborough RT"
            )
            
            code = st.text_input(
                "Incident Code",
                value="SUO",
                help="Delay reason code (e.g., MUSC, TUSPD, EUNT, SUO). If you don't know this just use SUO (Passenger Other)"
            )
            
            bound = st.selectbox(
                "Direction",
                ["N", "E", "S", "W"],
                help="N=North, E=East, S=South, W=West"
            )
        
        if st.button("Predict Delay", type="primary"):
            with st.spinner("Making prediction..."):
                input_data = {
                    'Date': str(input_date),
                    'Time': input_time.strftime('%H:%M'),
                    'Station': station.upper(),
                    'Line': line,
                    'Code': code.upper(),
                    'Bound': bound
                }
                
                result = predictor.predict(input_data)
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                # Display prediction
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    prediction_class = "delay-yes" if result['prediction'] == 'Delay' else "delay-no"
                    st.markdown(f"### Prediction")
                    st.markdown(f"<div class='big-metric {prediction_class}'>{result['prediction']}</div>", 
                              unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"### Delay Probability")
                    st.markdown(f"<div class='big-metric'>{result['delay_probability']:.1%}</div>", 
                              unsafe_allow_html=True)
                
                with col_c:
                    st.markdown(f"### No Delay Probability")
                    st.markdown(f"<div class='big-metric'>{result['no_delay_probability']:.1%}</div>", 
                              unsafe_allow_html=True)
                
                # Probability bar chart
                st.markdown("---")
                fig, ax = plt.subplots(figsize=(8, 2))
                categories = ['No Delay', 'Delay']
                probabilities = [result['no_delay_probability'], result['delay_probability']]
                colors = ['#2ca02c', '#d62728']
                
                ax.barh(categories, probabilities, color=colors, alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Confidence')
                for i, v in enumerate(probabilities):
                    ax.text(v + 0.02, i, f'{v:.1%}', va='center')
                
                st.pyplot(fig)
                
                # Show input details
                with st.expander("📋 Input Details"):
                    st.json(input_data)
    
    # TAB 2: 24-Hour Forecast
    with tab2:
        st.header("24-Hour Delay Forecast")
        st.markdown("Predict delay probabilities for every hour of a specific day")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_date = st.date_input(
                "Select Date",
                value=date.today(),
                min_value=date(2018, 1, 1),
                max_value=date(2030, 12, 31),
                key="forecast_date"
            )
            
            forecast_station = st.text_input(
                "Station",
                value="BLOOR YONGE STATION",
                key="forecast_station"
            )
        
        with col2:
            forecast_line = st.selectbox(
                "Line",
                ["YU", "BD", "SHP", "SRT"],
                key="forecast_line"
            )
            
            forecast_code = st.text_input(
                "Incident Code",
                value="SUO",
                key="forecast_code"
            )
            
            forecast_bound = st.selectbox(
                "Direction",
                ["N", "E", "S", "W"],
                key="forecast_bound"
            )
        
        if st.button("📊 Generate 24-Hour Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                try:
                    result = predictor.predict_day_hourly(
                        target_date=str(forecast_date),
                        station=forecast_station.upper(),
                        line=forecast_line,
                        code=forecast_code.upper(),
                        bound=forecast_bound
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(result)
                    
                    st.markdown("---")
                    st.subheader("Hourly Delay Probabilities")
                    
                    # Line chart
                    fig, ax = plt.subplots(figsize=(14, 5))
                    
                    ax.plot(df['hour'], df['delay_probability'], 
                           marker='o', linewidth=2, markersize=6, color='#1f77b4')
                    ax.fill_between(df['hour'], df['delay_probability'], 
                                   alpha=0.3, color='#1f77b4')
                    
                    # Highlight rush hours
                    rush_hours = [7, 8, 9, 17, 18]
                    for hour in rush_hours:
                        ax.axvspan(hour-0.5, hour+0.5, alpha=0.1, color='red')
                    
                    ax.set_xlabel('Hour of Day', fontsize=12)
                    ax.set_ylabel('Delay Probability', fontsize=12)
                    ax.set_title(f'Delay Probability Throughout the Day - {forecast_date}', 
                               fontsize=14, fontweight='bold')
                    ax.set_xticks(range(0, 24))
                    ax.set_ylim(0, 100)
                    ax.grid(True, alpha=0.3)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
                    
                    # Legend
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor='red', alpha=0.1, label='Rush Hours')]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    st.pyplot(fig)
                    
                    # Stats
                    st.markdown("---")
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    
                    with col_s1:
                        st.metric("Avg Delay Probability", f"{df['delay_probability'].mean():.1f}%")
                    
                    with col_s2:
                        peak_hour = df.loc[df['delay_probability'].idxmax()]
                        st.metric("Peak Delay Hour", f"{int(peak_hour['hour'])}:00")
                    
                    with col_s3:
                        st.metric("Max Probability", f"{df['delay_probability'].max():.1f}%")
                    
                    with col_s4:
                        st.metric("Min Probability", f"{df['delay_probability'].min():.1f}%")
                    
                    # Data table
                    with st.expander("📋 View Hourly Data"):
                        display_df = df.copy()
                        display_df['hour'] = display_df['hour'].apply(lambda x: f"{int(x):02d}:00")
                        display_df['delay_probability'] = display_df['delay_probability'].apply(lambda x: f"{x:.1f}%")
                        display_df['prediction'] = display_df['prediction'].apply(
                            lambda x: "🔴 Delay" if x == "Delay" else "🟢 No Delay"
                        )
                        st.dataframe(display_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Built using Streamlit and scikit-learn</p>
            <p>Model trained on TTC delay data (2018-2025)</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
