import numpy as np
import pandas as pd

def feature_engineering():

    # SubwayDelays.csv
    subway_df = pd.read_csv('data/final/final.csv')

    subway_df['Date'] = pd.to_datetime(subway_df['Date'])
    subway_df['DayOfWeek'] = subway_df['Date'].dt.dayofweek
    subway_df['Month'] = subway_df['Date'].dt.month
    subway_df['Year'] = subway_df['Date'].dt.year
    subway_df['Hour'] = pd.to_datetime(subway_df['Time'], format='%H:%M', errors='coerce').dt.hour
    subway_df['IsWeekend'] = (subway_df['DayOfWeek'] >= 5).astype(int)
    subway_df['IsRushHour'] = subway_df['Hour'].isin([7, 8, 9, 17, 18]).astype(int)

    subway_df['HasDelay'] = (subway_df['Min Delay'] > 0).astype(int) # has a delay if the length of the delay is greater than 0

    subway_df['HourBin'] = pd.cut(subway_df['Hour'], bins=[0, 6, 10, 16, 19, 24], labels=['Night', 'Morning', 'Midday', 'Evening', 'Late'], include_lowest=True) # feature grouping hours by time of day
    subway_df['RushHour_Weekday'] = subway_df['IsRushHour'] * (1 - subway_df['IsWeekend']) # feature, if rushhour + weekend
    subway_df['Weekend_Morning'] = subway_df['IsWeekend'] * (subway_df['Hour'] < 12).astype(int) # feature, if weekend + morning
    subway_df['Season'] = subway_df['Month'].apply(lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3) # feature grouping months by seasons

    # CYCLICAL ENCODING (so model knows hour 23 is close to hour 0)
    subway_df['Hour_sin'] = np.sin(2 * np.pi * subway_df['Hour'] / 24)
    subway_df['Hour_cos'] = np.cos(2 * np.pi * subway_df['Hour'] / 24)
    subway_df['DayOfWeek_sin'] = np.sin(2 * np.pi * subway_df['DayOfWeek'] / 7)
    subway_df['DayOfWeek_cos'] = np.cos(2 * np.pi * subway_df['DayOfWeek'] / 7) # basically plots hours and days of week relative to sin/cos, so the ends meet and are less "different".

    hour_delay_rate = subway_df.groupby('Hour')['HasDelay'].mean()
    subway_df['Hour_DelayRate'] = subway_df['Hour'].map(hour_delay_rate) # mean of delay per hour

    day_delay_rate = subway_df.groupby('DayOfWeek')['HasDelay'].mean()
    subway_df['Day_DelayRate'] = subway_df['DayOfWeek'].map(day_delay_rate) # mean of delay per day of week

    station_delay_rate = subway_df.groupby('Station')['HasDelay'].mean()
    subway_df['Station_DelayRate'] = subway_df['Station'].map(station_delay_rate) # mean of delay per station

    line_delay_rate = subway_df.groupby('Line')['HasDelay'].mean()
    subway_df['Line_DelayRate'] = subway_df['Line'].map(line_delay_rate) # ... etc

    code_delay_rate = subway_df.groupby('Code')['HasDelay'].mean()
    subway_df['Code_DelayRate'] = subway_df['Code'].map(code_delay_rate)

    # CATEGORICAL FEATURES - One-hot encoding
    # Get top 10 most common stations to avoid too many features, common stations are subject to more delays!
    top_stations = subway_df['Station'].value_counts().head(10).index
    subway_df['Station_Category'] = subway_df['Station'].apply(
        lambda x: x if x in top_stations else 'Other'
    )

    subway_df = pd.get_dummies(subway_df, columns=['HourBin'], prefix='Time') #one hot encoding for time feature

    delay_rates = {
        'hour': hour_delay_rate.to_dict(),
        'day': day_delay_rate.to_dict(),
        'station': station_delay_rate.to_dict(),
        'line': line_delay_rate.to_dict(),
        'code': code_delay_rate.to_dict(),
    }

    return subway_df, delay_rates