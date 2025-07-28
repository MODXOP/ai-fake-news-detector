from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
import os
from datetime import datetime

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html", prediction=None, confidence=None, input_text="")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    vect_news = vectorizer.transform([news])

    pred_proba = model.predict_proba(vect_news)[0]
    pred = model.predict(vect_news)[0]
    confidence = round(np.max(pred_proba) * 100, 2)
    label = "Fake" if pred == 0 else "Real"

    # Save to log CSV
    log_file = 'prediction_log.csv'
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Input Text', 'Prediction', 'Confidence'])
        writer.writerow([datetime.now(), news, label, confidence])

    return render_template("index.html", prediction=label, confidence=confidence, input_text=news)

@app.route("/history")
def history():
    log_file = 'prediction_log.csv'
    rows = []
    if os.path.exists(log_file):
        with open(log_file, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)
            rows = list(reader)
    else:
        headers = ["Timestamp", "Input Text", "Prediction", "Confidence"]
    return render_template("history.html", headers=headers, rows=rows)

if __name__ == "__main__":
    app.run(debug=True)
