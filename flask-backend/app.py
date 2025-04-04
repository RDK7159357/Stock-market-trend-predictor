from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)
model = load_model("lstm_model.h5")
with open("scalers.pkl", "rb") as file:
    scalers = pickle.load(file)

def predict_trends(num_days):
    # Dummy prediction function – replace with your prediction logic.
    today = datetime.today()
    dates = [(today + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(num_days)]
    prices = np.linspace(100, 200, num=num_days).tolist()
    trends = ['Up' if i % 2 == 0 else 'Down' for i in range(num_days)]
    return dates, prices, trends

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    days = int(data.get("days", 20))
    dates, prices, trends = predict_trends(days)
    return jsonify({"dates": dates, "prices": prices, "trends": trends})

if __name__ == "__main__":
    app.run(debug=True)
