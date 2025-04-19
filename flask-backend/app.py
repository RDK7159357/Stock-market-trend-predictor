from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and scaler
model = load_model("model2/lstm_stock_model.h5")
scaler = joblib.load("model2/scaler.pkl")


def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    if 'Date' not in df.columns:
        raise ValueError("Date column not found in the dataset.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Load historical data
original_df = load_csv("dataset/Amazon_large.csv")
scaled_data = scaler.transform(original_df[['Price', 'High', 'Low', 'Open', 'Volume']])

def predict_prices_for_date_range(start_date, end_date, n_steps=30):
    # Calculate the number of days in the date range
    date_range = pd.date_range(start=start_date, end=end_date)
    future_steps = len(date_range)
    
    # Ensure scaled_data has at least n_steps
    if len(scaled_data) < n_steps:
        raise ValueError(f"scaled_data must have at least {n_steps} data points")
    
    # Use the last n_steps of historical data to start predictions
    input_data = scaled_data[-n_steps:].reshape((1, n_steps, scaled_data.shape[1]))
    
    future_prices = []
    
    for _ in range(future_steps):
        # Predict next step
        pred = model.predict(input_data)
        future_prices.append(pred[0, 0])  # Append predicted price
        
        # Update input data to include the new prediction
        input_data = np.roll(input_data, -1, axis=1)
        input_data[0, -1, 0] = pred  # Update the price column
    
    # Inverse-transform predictions to original scale
    pred_full = np.zeros((future_steps, scaled_data.shape[1]))
    pred_full[:, 0] = future_prices
    future_prices_actual = scaler.inverse_transform(pred_full)[:, 0]
    
    # Generate trend data
    trends = []
    for i in range(len(future_prices_actual)):
        if i == 0:
            trends.append("Up")  # Default for the first day
        else:
            if future_prices_actual[i] > future_prices_actual[i-1]:
                trends.append("Up")
            else:
                trends.append("Down")
    
    # Format dates
    dates = [date.strftime('%Y-%m-%d') for date in date_range]
    
    return dates, future_prices_actual.tolist(), trends

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")
        
        if not start_date_str or not end_date_str:
            return jsonify({"error": "Missing start_date or end_date in request"}), 400
        
        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # Make predictions for the specified date range
        dates, prices, trends = predict_prices_for_date_range(start_date, end_date)
        
        return jsonify({"dates": dates, "prices": prices, "trends": trends})
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)