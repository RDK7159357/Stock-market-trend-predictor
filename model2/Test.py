from keras.models import load_model
import joblib
import pandas as pd
import numpy as np

# Define the load_csv function
def load_csv(file_path):
    df = pd.read_csv(file_path)
    # Ensure columns are correctly formatted
    df.columns = df.columns.str.strip()
    if 'Date' not in df.columns:
        raise ValueError("Date column not found in the dataset.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Load the saved model
loaded_model = load_model('./model2/lstm_stock_model.h5')

# Load the saved scaler
loaded_scaler = joblib.load('./model2/scaler.pkl')

def predict_prices_for_date_range(model, scaler, last_known_data, start_date, end_date, n_steps=30):
    """
    Predict future stock prices for a specific date range.
    Args:
        model: Trained LSTM model
        scaler: Pre-trained MinMaxScaler
        last_known_data: Scaled historical data (last n_steps or more)
        start_date: Start date for predictions (datetime object)
        end_date: End date for predictions (datetime object)
        n_steps: Number of historical steps to use for prediction
    Returns:
        Predicted future prices (inverse-scaled) and corresponding dates
    """
    # Ensure last_known_data has at least n_steps
    if len(last_known_data) < n_steps:
        raise ValueError(f"last_known_data must have at least {n_steps} data points")

    # Use the last n_steps of historical data to start predictions
    input_data = last_known_data[-n_steps:].reshape((1, n_steps, last_known_data.shape[1]))
    
    # Generate date range for predictions
    date_range = pd.date_range(start=start_date, end=end_date)
    future_steps = len(date_range)
    
    future_prices = []
    
    for _ in range(future_steps):
        # Predict next step
        pred = model.predict(input_data)
        future_prices.append(pred[0, 0])  # Append predicted price
        
        # Update input data to include the new prediction
        input_data = np.roll(input_data, -1, axis=1)
        input_data[0, -1, 0] = pred  # Update the price column
    
    # Inverse-transform predictions to original scale
    pred_full = np.zeros((future_steps, last_known_data.shape[1]))
    pred_full[:, 0] = future_prices
    future_prices_actual = scaler.inverse_transform(pred_full)[:, 0]
    
    # Combine dates and predictions
    predictions_with_dates = list(zip(date_range, future_prices_actual))
    
    # Print each prediction with date
    for date, price in predictions_with_dates:
        print(f"Predicted Price for {date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    return predictions_with_dates

# Load the original dataset to get the scaled data
original_df = load_csv('./dataset/Amazon_large.csv')
scaled_data = loaded_scaler.transform(original_df[['Price', 'High', 'Low', 'Open', 'Volume']])

# Specify the date range for predictions
start_date = pd.to_datetime('2025-01-01')
end_date = pd.to_datetime('2026-01-10')

# Make predictions for the specified date range
future_predictions = predict_prices_for_date_range(
    loaded_model, 
    loaded_scaler, 
    scaled_data, 
    start_date=start_date, 
    end_date=end_date
)