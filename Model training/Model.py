import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle

def main():
    # Get user input for prediction days
    while True:
        try:
            days_to_predict = int(input("Enter the number of days to predict (1-365): "))
            if 1 <= days_to_predict <= 365:
                break
            else:
                print("Please enter a number between 1 and 365.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"\nPredicting stock trends for the next {days_to_predict} days...\n")

    # Load and preprocess data
    df = pd.read_csv("./dataset/Amazon.csv")
    df = df.drop([0, 1])
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df.reset_index(drop=True, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Close', 'High', 'Low', 'Open']] = df[['Close', 'High', 'Low', 'Open']].astype(float)
    df['Volume'] = df['Volume'].astype(int)
    df.set_index('Date', inplace=True)

    # Plot historical data
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Amazon Closing Price')
    plt.title('Amazon Stock Closing Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('historical_price.png')
    plt.close()

    # Extract seasonality features
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['Quarter'] = df.index.quarter

    # Current date for prediction reference
    current_date = datetime.today()
    
    # Define the target prediction period
    start_predict_date = current_date
    end_predict_date = current_date + timedelta(days=days_to_predict)
    
    # Function to get similar seasonal data from previous years
    def get_seasonal_training_data(df, current_date, days_to_predict, years_back=5):
        """
        Extract training data from similar time periods in previous years
        """
        target_month = current_date.month
        target_day = current_date.day
        
        # Calculate the date range for prediction
        start_date = current_date
        end_date = current_date + timedelta(days=days_to_predict)
        
        # Initialize dataframe to store seasonal training data
        seasonal_data = pd.DataFrame()
        
        # Get data from similar periods in previous years
        current_year = current_date.year
        for year in range(current_year - years_back, current_year):
            # Calculate similar period for this year
            year_start = datetime(year, target_month, target_day)
            year_end = year_start + timedelta(days=days_to_predict)
            
            # Adjust if date doesn't exist (like Feb 29)
            try:
                pd.Timestamp(year_start)
            except ValueError:
                year_start = datetime(year, target_month, target_day-1)
            
            # Get data for this period
            mask = (df.index >= year_start) & (df.index <= year_end)
            year_data = df.loc[mask].copy()
            
            # Add to seasonal data if not empty
            if not year_data.empty:
                # Add a year identifier for visualization
                year_data['PredictionYear'] = year
                seasonal_data = pd.concat([seasonal_data, year_data])
        
        # Also include some recent data (last 30 days)
        recent_end = df.index.max()
        recent_start = recent_end - timedelta(days=30)
        recent_data = df.loc[(df.index >= recent_start) & (df.index <= recent_end)].copy()
        recent_data['PredictionYear'] = 'Recent'
        
        # Combine seasonal and recent data
        training_data = pd.concat([seasonal_data, recent_data])
        
        return training_data
    
    # Get seasonal training data
    seasonal_training_data = get_seasonal_training_data(df, current_date, days_to_predict)
    
    # Visualize seasonal patterns
    plt.figure(figsize=(14, 8))
    for year, group in seasonal_training_data.groupby('PredictionYear'):
        if year == 'Recent':
            plt.plot(group.index, group['Close'], label=f'Recent Data', linewidth=3)
        else:
            plt.plot(group.index, group['Close'], label=f'Similar Period {year}')
    
    plt.title(f'Amazon Stock Price Patterns for Similar Periods (April-May)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('seasonal_patterns.png')
    plt.close()

    # Prepare data for modeling
    # Features: Close, Month, Day, DayOfWeek, Quarter
    features = ['Close', 'Month', 'Day', 'DayOfWeek', 'Quarter']
    
    # Scale the features
    scaler_dict = {}
    scaled_data = pd.DataFrame(index=seasonal_training_data.index)
    
    for feature in features:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[feature] = scaler.fit_transform(seasonal_training_data[feature].values.reshape(-1, 1)).flatten()
        scaler_dict[feature] = scaler
    
    # Function to create sequences with multiple features
    def create_multivariate_sequences(data, features, time_step=30):
        X, y = [], []
        for i in range(time_step, len(data)):
            # For each feature, get the time_step previous values
            feature_sequences = []
            for feature in features:
                feature_seq = data[feature].iloc[i-time_step:i].values
                feature_sequences.append(feature_seq)
            
            # Stack the feature sequences
            X_seq = np.column_stack(feature_sequences)
            X.append(X_seq)
            
            # Target is the next Close price
            y.append(data['Close'].iloc[i])
        
        return np.array(X), np.array(y)
    
    # Create sequences with a shorter lookback period (30 days instead of 60)
    X, y = create_multivariate_sequences(scaled_data, features, time_step=30)
    
    # Reshape X to be 3D: [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], len(features))
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train the model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Training LSTM model... (This may take a few minutes)")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    print("Model training complete!\n")

    # Evaluate on test data
    test_predictions = model.predict(X_test)
    
    # Inverse transform the predictions and actual values
    predicted_prices = scaler_dict['Close'].inverse_transform(test_predictions.reshape(-1, 1))
    actual_prices = scaler_dict['Close'].inverse_transform(y_test.reshape(-1, 1))

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Price')
    plt.plot(predicted_prices, label='Predicted Price')
    plt.title('Amazon Stock Price: Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    # Define future prediction function with season-aware trend analysis
    def predict_future_seasonal(model, scaler_dict, features, df, current_date, days):
        """
        Predict future stock prices and trends using seasonal patterns
        Args:
            model: Trained LSTM model
            scaler_dict: Dictionary of fitted MinMaxScalers for each feature
            features: List of feature names
            df: Original dataframe with historical data
            current_date: Current date for prediction
            days: Number of days to predict
        Returns:
            future_dates, future_prices, future_trends, trend_directions
        """
        # Generate future dates (only business days)
        future_dates = []
        current_date_copy = current_date
        
        while len(future_dates) < days:
            current_date_copy += timedelta(days=1)
            # Skip weekends (this is a simple approach, doesn't account for holidays)
            if current_date_copy.weekday() < 5:  # Monday to Friday
                future_dates.append(current_date_copy)
        
        # Get the last available sequence of data (30 days)
        # We'll try to find the latest data that ends before the current date
        latest_data = df[df.index < current_date].tail(30)
        
        if len(latest_data) < 30:
            # If we don't have enough data, take whatever we have
            print("Warning: Using less than 30 days of data for initial sequence")
            
        # Create a dataframe for feature engineering
        last_sequence_df = latest_data.copy()
        last_sequence_df['Month'] = last_sequence_df.index.month
        last_sequence_df['Day'] = last_sequence_df.index.day
        last_sequence_df['DayOfWeek'] = last_sequence_df.index.dayofweek
        last_sequence_df['Quarter'] = last_sequence_df.index.quarter
        
        # Scale the features
        scaled_seq = {}
        for feature in features:
            scaled_seq[feature] = scaler_dict[feature].transform(last_sequence_df[feature].values.reshape(-1, 1)).flatten()
        
        # Create initial input for prediction
        input_data = np.column_stack([scaled_seq[feature] for feature in features])
        
        # Make predictions for specified number of days
        future_prices = []
        input_sequence = input_data[-30:].copy()  # Use last 30 days as initial sequence
        
        for date in future_dates:
            # Create seasonal features for this future date
            future_month = date.month
            future_day = date.day
            future_dayofweek = date.weekday()
            future_quarter = (date.month - 1) // 3 + 1
            
            # Scale the seasonal features
            scaled_month = scaler_dict['Month'].transform([[future_month]])[0][0]
            scaled_day = scaler_dict['Day'].transform([[future_day]])[0][0]
            scaled_dayofweek = scaler_dict['DayOfWeek'].transform([[future_dayofweek]])[0][0]
            scaled_quarter = scaler_dict['Quarter'].transform([[future_quarter]])[0][0]
            
            # Prepare input for model (shape: [1, 30, 5])
            x_input = input_sequence.reshape(1, 30, 5)
            
            # Get prediction for next day
            next_pred = model.predict(x_input, verbose=0)
            
            # Scale back the prediction
            next_price = scaler_dict['Close'].inverse_transform(next_pred.reshape(-1, 1))[0][0]
            future_prices.append(next_price)
            
            # Update input sequence for next prediction
            # Remove first row and add new prediction with seasonal features
            input_sequence = np.delete(input_sequence, 0, axis=0)
            new_row = np.array([[next_pred[0][0], scaled_month, scaled_day, scaled_dayofweek, scaled_quarter]])
            input_sequence = np.vstack([input_sequence, new_row])
        
        # Calculate trends (1 for up, 0 for down, compared to previous day)
        future_trends = []
        
        # First day trend is compared to the last actual price
        last_actual_price = df[df.index < current_date]['Close'].iloc[-1]
        
        # Initialize with first predicted price compared to last actual price
        future_trends.append(1 if future_prices[0] > last_actual_price else 0)
        
        # Continue with remaining predictions
        for i in range(1, len(future_prices)):
            trend = 1 if future_prices[i] > future_prices[i-1] else 0
            future_trends.append(trend)
        
        # Convert binary trends to text directions
        trend_directions = ["Up" if trend == 1 else "Down" for trend in future_trends]
        
        return future_dates, future_prices, future_trends, trend_directions
    
    # Predict future prices and trends starting from the current date
    print(f"Generating {days_to_predict}-day prediction starting from {current_date.strftime('%Y-%m-%d')}...")
    future_dates, future_prices, future_trends, trend_directions = predict_future_seasonal(
        model, scaler_dict, features, df, current_date, days=days_to_predict
    )
    
    # Create a dataframe for future predictions
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_prices,
        'Trend': future_trends,
        'Direction': trend_directions
    })

    # Print the future trend predictions
    print("\nFuture Stock Trend Predictions:")
    print("================================")
    for i, row in future_df.iterrows():
        print(f"Day {i+1} ({row['Date'].strftime('%Y-%m-%d')}): ${row['Predicted_Price']:.2f} - {row['Direction']}")

    # Visualize future prices with trend indicators
    plt.figure(figsize=(14, 8))
    
    # Last actual price from dataset
    last_date = df[df.index < current_date].index.max()
    last_price = df.loc[last_date, 'Close']
    
    # Add the last actual price point to connect with predictions
    plt.plot([last_date], [last_price], 'bo', markersize=8, label='Last Actual Price')
    
    # Plot the predicted prices
    plt.plot(future_df['Date'], future_df['Predicted_Price'], label='Predicted Price', color='blue')
    
    # Add markers for up and down trends
    up_days = future_df[future_df['Trend'] == 1]
    down_days = future_df[future_df['Trend'] == 0]
    
    plt.scatter(up_days['Date'], up_days['Predicted_Price'], color='green', marker='^',
                s=100, label='Upward Trend', zorder=5)
    plt.scatter(down_days['Date'], down_days['Predicted_Price'], color='red', marker='v',
                s=100, label='Downward Trend', zorder=5)
    
    # Format the plot
    plt.title(f'Amazon Stock Price and Trend Prediction for Next {days_to_predict} Days (Starting {current_date.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Adjust date locator based on prediction length
    if days_to_predict <= 30:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_to_predict // 10)))
    elif days_to_predict <= 90:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Weekly
    else:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  # Monthly
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    output_filename = f'future_trends_{days_to_predict}_days.png'
    plt.savefig(output_filename)
    # plt.show()

    # Create a trend summary
    up_trend_count = sum(future_trends)
    down_trend_count = len(future_trends) - up_trend_count
    
    print(f"\nTrend Summary for Next {days_to_predict} Days:")
    print(f"Upward Trends: {up_trend_count} days ({up_trend_count/days_to_predict*100:.1f}%)")
    print(f"Downward Trends: {down_trend_count} days ({down_trend_count/days_to_predict*100:.1f}%)")
    
    # Calculate overall direction and average daily change
    total_price_change = future_prices[-1] - last_price
    avg_daily_change = total_price_change / days_to_predict
    overall_direction = "Upward" if total_price_change > 0 else "Downward"
    
    print(f"\nOverall {days_to_predict}-Day Trend: {overall_direction}")
    print(f"Starting Price (Last Actual): ${last_price:.2f}")
    print(f"Ending Price (Predicted): ${future_prices[-1]:.2f}")
    print(f"Total Price Change: ${total_price_change:.2f}")
    print(f"Average Daily Change: ${avg_daily_change:.2f}")
    
    # Calculate model evaluation metrics
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    print("\nModel Performance Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Convert to trends for historical data
    def get_trend(prices):
        return [1 if prices[i] > prices[i - 1] else 0 for i in range(1, len(prices))]
    
    actual_trend = get_trend(actual_prices.flatten())
    predicted_trend = get_trend(predicted_prices.flatten())
    
    # Evaluate classification metrics
    acc = accuracy_score(actual_trend, predicted_trend)
    prec = precision_score(actual_trend, predicted_trend)
    rec = recall_score(actual_trend, predicted_trend)
    f1 = f1_score(actual_trend, predicted_trend)
    
    print(f"Trend Accuracy: {acc:.4f}")
    print(f"Trend Precision: {prec:.4f}")
    print(f"Trend Recall: {rec:.4f}")
    print(f"Trend F1-score: {f1:.4f}")

    # Plot a comprehensive visualization with historical and future data
    plt.figure(figsize=(16, 8))
    
    # Plot some historical data (last 60 days before current date)
    historical_start = current_date - timedelta(days=60)
    historical_data = df[(df.index >= historical_start) & (df.index < current_date)]
    plt.plot(historical_data.index, historical_data['Close'], 
             label='Historical Price', color='blue')
    
    # Plot the future predicted prices
    plt.plot(future_df['Date'], future_df['Predicted_Price'], 
             label='Predicted Price', color='orange', linestyle='--')
    
    # Add a vertical line to separate historical and future data
    plt.axvline(x=current_date, color='black', linestyle='-', alpha=0.7)
    plt.text(current_date, plt.ylim()[0], 'Today', horizontalalignment='center', verticalalignment='bottom')
    
    # Add markers for up and down trends
    plt.scatter(up_days['Date'], up_days['Predicted_Price'], color='green', marker='^',
                s=100, label='Predicted Upward Trend', zorder=5)
    plt.scatter(down_days['Date'], down_days['Predicted_Price'], color='red', marker='v',
                s=100, label='Predicted Downward Trend', zorder=5)
    
    # Format the plot
    plt.title(f'Amazon Stock: Historical Data and {days_to_predict}-Day Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    comprehensive_filename = f'comprehensive_forecast_{days_to_predict}_days.png'
    plt.savefig(comprehensive_filename)
    
    # Save results to CSV
    csv_filename = f'future_predictions_{days_to_predict}_days.csv'
    future_df.to_csv(csv_filename, index=False)
    
    # Save model and scalers
    model.save("lstm_model.h5")
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scaler_dict, f)
        
    print(f"\nResults saved to '{csv_filename}'")
    print(f"Visualizations saved to '{output_filename}' and '{comprehensive_filename}'")
    print(f"Seasonal patterns visualization saved to 'seasonal_patterns.png'")
    print("Model saved as 'lstm_model.h5'")
    print("Scalers saved as 'scalers.pkl'")

if __name__ == "__main__":
    main()