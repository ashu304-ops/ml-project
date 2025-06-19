from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    freq = float(request.form['frequency'])
    amp = float(request.form['amplitude'])
    rms = float(request.form['rms'])

    features = np.array([[freq, amp, rms]])
    pred = model.predict(features)[0]
    result = "Healthy" if pred == 0 else "Needs Maintenance"

    return f"<h2>Prediction Result: {result}</h2><br><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

