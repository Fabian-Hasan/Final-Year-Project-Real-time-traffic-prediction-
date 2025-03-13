import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='preprocessing.log'
)

def load_and_validate_data(file_path):
    """Load and perform initial validation of the traffic data."""
    try:
        data = pd.read_csv(file_path, low_memory=False)
        logging.info(f"Successfully loaded dataset with shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def preprocess_traffic_data(data):
    """Preprocess traffic data for machine learning models."""
    try:
        # Convert timestamps
        data['count_date'] = pd.to_datetime(data['count_date'])
        data['timestamp'] = data['count_date'] + pd.to_timedelta(data['hour'], unit='h')
        
        # Create features
        vehicle_columns = [
            'pedal_cycles', 'two_wheeled_motor_vehicles', 'cars_and_taxis',
            'buses_and_coaches', 'LGVs', 'HGVs_2_rigid_axle', 'HGVs_3_rigid_axle',
            'HGVs_4_or_more_rigid_axle', 'HGVs_3_or_4_articulated_axle',
            'HGVs_5_articulated_axle', 'HGVs_6_articulated_axle'
        ]
        
        # Fill missing values
        data[vehicle_columns] = data[vehicle_columns].fillna(0)
        
        # Create time-based features
        data['hour_of_day'] = data['hour']
        data['day_of_week'] = data['count_date'].dt.dayofweek
        data['month'] = data['count_date'].dt.month
        data['is_weekend'] = data['count_date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Calculate total vehicles (excluding pedal cycles)
        motor_vehicle_columns = [col for col in vehicle_columns if col != 'pedal_cycles']
        data['total_motor_vehicles'] = data[motor_vehicle_columns].sum(axis=1)
        
        # Create location identifier
        data['location_id'] = data['road_name'] + "_" + data['direction_of_travel']
        
        # Select final features
        features = [
            'timestamp', 'location_id', 'total_motor_vehicles', 'hour_of_day',
            'day_of_week', 'month', 'is_weekend', 'latitude', 'longitude'
        ]
        processed_data = data[features].copy()
        
        return processed_data
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def normalize_and_save(processed_data, output_file, scaler_file):
    """Normalize the data and save both data and scaler."""
    try:
        # Set timestamp as index
        processed_data.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        processed_data.sort_index(inplace=True)
        
        # Normalize traffic counts
        scaler = MinMaxScaler()
        processed_data['normalized_traffic'] = scaler.fit_transform(
            processed_data[['total_motor_vehicles']]
        )
        
        # Save the scaler
        joblib.dump(scaler, scaler_file)
        logging.info(f"Saved scaler to {scaler_file}")
        
        # Save processed data
        processed_data.to_csv(output_file)
        logging.info(f"Saved processed data to {output_file}")
        
        return processed_data
        
    except Exception as e:
        logging.error(f"Error in normalization and saving: {str(e)}")
        raise

def main():
    """Main preprocessing pipeline."""
    try:
        # Update file paths to match your directory
        input_file = 'preprocessed_traffic.csv'  # Changed to match your existing file
        output_file = 'processed_traffic_final.csv'  # Changed name to avoid overwriting
        scaler_file = 'traffic_scaler.pkl'
        
        # Load data
        logging.info("Starting data preprocessing...")
        raw_data = load_and_validate_data(input_file)
        
        # Preprocess data
        processed_data = preprocess_traffic_data(raw_data)
        
        # Normalize and save
        final_data = normalize_and_save(processed_data, output_file, scaler_file)
        
        # Print summary statistics
        print("\nPreprocessing Summary:")
        print(f"Total samples: {len(final_data)}")
        print(f"Date range: {final_data.index.min()} to {final_data.index.max()}")
        print(f"Number of unique locations: {final_data['location_id'].nunique()}")
        print(f"Average daily traffic: {final_data['total_motor_vehicles'].mean():.2f}")
        
        logging.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
        raise

def load_data():
    # Load preprocessed data
    data = pd.read_csv('processed_traffic_final.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

def train_arima_model():
    # Load data
    data = load_data()
    
    # Select a single location for initial testing
    location = data['location_id'].unique()[0]
    location_data = data[data['location_id'] == location]['normalized_traffic']
    
    # Split data into train and test
    train_size = int(len(location_data) * 0.8)
    train, test = location_data[:train_size], location_data[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train, order=(1,1,1))  # You can tune these parameters
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(len(test))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)
    
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    
    # Save model
    joblib.dump(model_fit, 'arima_model.pkl')
    
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm_model():
    # Load data
    data = pd.read_csv('processed_traffic_final.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Select features
    features = ['normalized_traffic', 'hour_of_day', 'day_of_week', 'is_weekend']
    
    # Prepare sequences
    seq_length = 24  # Use 24 hours of data to predict the next hour
    X, y = create_sequences(data[features].values, seq_length)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, len(features))),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X_train, y_train, 
              epochs=10, 
              batch_size=32,
              validation_split=0.1,
              verbose=1)
    
    # Save model
    model.save('lstm_model.h5')
    
if __name__ == "__main__":
    main()
