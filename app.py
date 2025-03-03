from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

with open('malaria.model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    fever = float(data['fever'])
    headache = int(data['headache'])
    lack_of_appetites = int(data['lack_of_appetites'])
    body_pains = int(data['body_pains'])
    fatigue = int(data['fatigue'])

    input_data = scaler.transform([[fever, headache, lack_of_appetites, body_pains, fatigue]])

    prediction = model.predict(input_data)[0]
    result = "Positive for Malaria" if prediction == 1 else "Negative for Malaria"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)