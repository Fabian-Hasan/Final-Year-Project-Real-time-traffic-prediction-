# TrafficPredict Pro: Real-Time Traffic Prediction System

## What is TrafficPredict Pro?
TrafficPredict Pro is a system that uses **machine learning** to predict traffic conditions and suggest the best routes. It helps people save time, avoid traffic jams, and reduce fuel usage.

---

## What Can It Do?
1. **Traffic Prediction**:
   - Predicts traffic conditions using **ARIMA** and **LSTM** models.
   - Tells you how busy the roads will be in the next hour or 24 hours.

2. **Route Suggestions**:
   - Suggests the fastest or least congested routes based on traffic predictions.

3. **Interactive Maps and Charts**:
   - Shows traffic predictions and routes on **interactive maps** and **charts**.

4. **Performance Metrics**:
   - Displays how accurate the predictions are using **RMSE** and **MAE** metrics.

---

## How Does It Work?
- **Machine Learning**:
  - **ARIMA**: A statistical model for predicting traffic patterns.
  - **LSTM**: A deep learning model for more complex traffic predictions.

- **Web Application**:
  - Built using **Python** and **Flask**.
  - Users can enter start and end points to get traffic predictions and route suggestions.

- **Data Processing**:
  - Uses **Pandas** and **NumPy** to clean and organize traffic data.

- **Visualization**:
  - Uses **Plotly** for charts and **Leaflet.js** for maps.

---

## How to Set It Up

### What You Need
- **Python 3.7 or higher**
- **pip** (Python package installer)

### Steps to Run the Project
- go to https://www.data.gov.uk/dataset/208c0e7b-353f-4e2d-8b7a-1a7118467acc/gb-road-traffic-counts and download 'Raw Counts' and extract it and save to project folder
- pip install requirements.txt 
- python preprocess_data.py 
- python arima_model.py
- python lstm_model.py
- python app.py 
- Go to http://localhost:5000 to use the system
- Follow instructions on the screen 

### Main Structure
app.py: The main web application.

preprocess_data.py: Cleans and prepares the traffic data.

arima_model.py: Trains the ARIMA model.

lstm_model.py: Trains the LSTM model.

templates/: Contains the web pages (HTML files).

processed_traffic_final.csv: The cleaned traffic data.

arima_model.pkl: The trained ARIMA model.

lstm_model.h5: The trained LSTM model. 


