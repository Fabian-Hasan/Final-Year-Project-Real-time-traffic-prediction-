import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import joblib

def train_arima():
    # Load data
    print("Loading data for ARIMA...")
    data = pd.read_csv('processed_traffic_final.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Get one location's data for simplicity
    location_data = data[data['location_id'] == data['location_id'].unique()[0]]
    traffic_data = location_data['normalized_traffic']
    
    # Try different ARIMA parameters
    try:
        print("Training ARIMA model with parameters (1,1,1)...")
        model = ARIMA(traffic_data, order=(1,1,1))
        model_fit = model.fit()
    except:
        try:
            print("Trying with parameters (0,1,1)...")
            model = ARIMA(traffic_data, order=(0,1,1))
            model_fit = model.fit()
        except:
            print("Trying with parameters (1,0,1)...")
            model = ARIMA(traffic_data, order=(1,0,1))
            model_fit = model.fit()
    
    # Save model
    joblib.dump(model_fit, 'arima_model.pkl')
    print("ARIMA model saved as 'arima_model.pkl'")
    
    # Test prediction
    future_steps = model_fit.forecast(steps=24)  # Predict next 24 hours
    print("\nSample ARIMA prediction for next 24 hours:")
    print(future_steps[:5])  # Show first 5 predictions

if __name__ == "__main__":
    train_arima()