import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the CSV dataset
def load_csv(file_path):
    df = pd.read_csv(file_path)
    # Ensure columns are correctly formatted
    df.columns = df.columns.str.strip()
    if 'Date' not in df.columns:
        raise ValueError("Date column not found in the dataset.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Create sequences for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, 0])  # We are predicting the 'Price'
    return np.array(X), np.array(y)

# Load the dataset
df = load_csv('./dataset/Amazon_large.csv')

# Feature selection and normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Price', 'High', 'Low', 'Open', 'Volume']])

# Handle any potential NaN values
scaled_data = np.nan_to_num(scaled_data)

# Time-based train-test split
train_size = int(len(scaled_data) * 0.8)
Xtr, Xte = scaled_data[:train_size], scaled_data[train_size:]

# Create sequences for both sets
n_steps = 30  # Sequence length
Xtr, ytr = create_sequences(Xtr, n_steps)
Xte, yte = create_sequences(Xte, n_steps)

# Check for any NaN values in the sequences
if np.any(np.isnan(Xtr)) or np.any(np.isnan(ytr)):
    raise ValueError("Training data contains NaN values.")
if np.any(np.isnan(Xte)) or np.any(np.isnan(yte)):
    raise ValueError("Test data contains NaN values.")

# Build the LSTM model
model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(Xtr.shape[1], Xtr.shape[2])),
    Dropout(0.2),
    LSTM(100, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile the model with Adam optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(Xtr, ytr, epochs=100, batch_size=32, validation_data=(Xte, yte), callbacks=[early_stopping])

# Predict on the test set
y_pred = model.predict(Xte)

# Calculate regression metrics
rmse = np.sqrt(mean_squared_error(yte, y_pred))
mae = mean_absolute_error(yte, y_pred)
r2 = r2_score(yte, y_pred)

# Print regression metrics
print(f"Regression Metrics:")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Inverse-transform the entire scaled data to get actual stock prices
# Reshape yte and y_pred to include all 6 features (even though we only predict 'Price')
yte_full = np.zeros((len(yte), scaled_data.shape[1]))
y_pred_full = np.zeros((len(y_pred), scaled_data.shape[1]))
yte_full[:, 0] = yte
y_pred_full[:, 0] = y_pred.flatten()

yte_actual = scaler.inverse_transform(yte_full)[:, 0]  # Extract 'Price' column
y_pred_actual = scaler.inverse_transform(y_pred_full)[:, 0]  # Extract 'Price' column

# Create date index for the test set
date_test = df.index[-len(yte):]

# Create pandas Series for actual and predicted prices
yte_series = pd.Series(yte_actual, index=date_test, name='Actual Price')
y_pred_series = pd.Series(y_pred_actual, index=date_test, name='Predicted Price')

# Plotting the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(yte_series, label='Actual Price', color='blue')
plt.plot(y_pred_series, label='Predicted Price', color='red', alpha=0.7)
plt.title('Actual vs Predicted Stock Price (Actual Scale)')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Calculate moving average for predictions
window_size = 5  # Adjust this value based on your needs
y_pred_ma = y_pred_series.rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))
plt.plot(yte_series, label='Actual Price', color='blue')
plt.plot(y_pred_series, label='Predicted Price', color='red', alpha=0.3)
plt.plot(y_pred_ma, label='Predicted Price (Moving Average)', color='green')
plt.title('Actual vs Predicted Stock Price with Moving Average')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Create classification labels for trend prediction
yte_class = (yte_actual > np.roll(yte_actual, 1)).astype(int)[1:]  # 1 if price went up, 0 if price went down
y_pred_class = (y_pred_actual > np.roll(y_pred_actual, 1)).astype(int)[1:]  # 1 if predicted price goes up, 0 if goes down

# Calculate classification metrics
accuracy = accuracy_score(yte_class, y_pred_class)
precision = precision_score(yte_class, y_pred_class)
recall = recall_score(yte_class, y_pred_class)
f1 = f1_score(yte_class, y_pred_class)

# Print classification metrics
print(f"\nClassification Metrics (Trend Prediction):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save the trained model
model.save('lstm_stock_model.h5')

# Save the scaler (required for preprocessing new data)
import joblib
joblib.dump(scaler, 'scaler.pkl')