import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

def train_lstm():
    # Load data
    print("Loading data for LSTM...")
    data = pd.read_csv('processed_traffic_final.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Get one location's data
    location_data = data[data['location_id'] == data['location_id'].unique()[0]]
    traffic_data = location_data['normalized_traffic'].values
    
    print(f"Preparing sequences from {len(traffic_data)} data points...")
    
    # Prepare sequences (24 hours input -> 1 hour output)
    X, y = [], []
    for i in range(len(traffic_data) - 24):
        if i % 1000 == 0:  # Progress indicator
            print(f"Processing sequence {i}/{len(traffic_data)}")
        X.append(traffic_data[i:i+24])
        y.append(traffic_data[i+24])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print(f"Final data shape: {X.shape}")
    
    # Create and train model with reduced complexity
    print("Training LSTM model...")
    model = Sequential([
        LSTM(32, input_shape=(24, 1)),  # Reduced units from 50 to 32
        Dense(1)
    ])
    
    # Compile with specific loss and metrics
    model.compile(optimizer='adam',
                 loss=MeanSquaredError(),
                 metrics=[MeanAbsoluteError()])
    
    # Train with early stopping
    history = model.fit(X, y, 
                       epochs=3,  # Reduced epochs
                       batch_size=64,  # Increased batch size
                       validation_split=0.1,
                       verbose=1)
    
    # Save model
    model.save('lstm_model.h5')
    print("LSTM model saved as 'lstm_model.h5'")
    
    # Test prediction
    test_input = X[-1:]
    prediction = model.predict(test_input)
    print("\nSample LSTM prediction:")
    print(prediction[0][0])

if __name__ == "__main__":
    # Set memory growth for GPU if available
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        print("No GPU available. Using CPU.")
    
    train_lstm()