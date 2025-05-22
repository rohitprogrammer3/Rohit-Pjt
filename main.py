from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    prediction = model.predict(std_data)

    result = 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
