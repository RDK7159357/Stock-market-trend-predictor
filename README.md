# Amazon Stock Predictor 📈

<img width="383" alt="Screenshot 2025-04-04 at 8 43 47 PM" src="https://github.com/user-attachments/assets/8de338c0-95fe-4e64-9b63-b1ba759766cb" />
<img width="383" alt="Screenshot 2025-04-04 at 8 44 22 PM" src="https://github.com/user-attachments/assets/597f7bee-fd71-4133-aae7-22fe1dd92f72" />
<img width="383" alt="Screenshot 2025-04-04 at 8 44 42 PM" src="https://github.com/user-attachments/assets/2a080593-1610-4ba8-9010-067034f03f29" />


A full-stack application for predicting Amazon stock prices using LSTM models with Flask backend and React Native mobile interface, plus PowerBI analytics.

## Project Structure 🗂️

```bash
.
├── Model training/            # Python model development
│   └── Model.py              # LSTM training script
├── PowerBI visualization/     # Data analytics
│   └── Stockanalysisbi.pbix  # PowerBI dashboard
├── ReactNative/               # Mobile application
│   └── Stocky/               # React Native project
├── assets/                    # Visualization outputs
│   ├── actual_vs_predicted.png
│   ├── comprehensive_forecast_20_days.png
│   ├── future_predictions_20_days.csv
│   ├── future_trends_20_days.png
│   ├── historical_price.png
│   └── seasonal_patterns.png
├── dataset/                   # Stock data
│   └── Amazon.csv            # Historical price data
├── flask-backend/             # Prediction API
│   └── app.py                # Flask server
├── models/                    # Trained models
│   ├── lstm_model.h5         # LSTM model weights
│   └── scalers.pkl           # Data scalers
├── ppt/                       # Presentations
│   └── Amazon-Stock-Predictor-A-Python-and-React-App.pptx
└── README.md                  # This file

Key Features ✨

LSTM Prediction Model - Accurate stock forecasts
Flask REST API - Robust backend service
React Native Mobile App - Cross-platform interface
PowerBI Dashboard - Advanced analytics visualization
Comprehensive Reporting - PNG/CSV outputs

Open PowerBI dashboard:
Launch PowerBI visualization/Stockanalysisbi.pbix

API Endpoints 🔌

Endpoint	Method	Description	Example Request
/predict	POST	Get stock predictions	{"days": 20}

Dependencies 📦

Backend:


Python 3.8+
TensorFlow 2.x
Flask
pandas, numpy


Frontend:

React Native
react-native-chart-kit
axios

