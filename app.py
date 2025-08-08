from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the ML model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    test_result = float(request.form['test_result'])
    length_of_stay = float(request.form['length_of_stay'])
    admission_frequency = int(request.form['admission_frequency'])

    input_data = np.array([[test_result, length_of_stay, admission_frequency]])
    prediction = model.predict(input_data)[0]

    result = "High Risk of Readmission" if prediction == 1 else "Low Risk of Readmission"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
