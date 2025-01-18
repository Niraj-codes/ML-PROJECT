from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the trained model
with open('california_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    input_features = np.array([data['features']])  # Convert input to numpy array
    prediction = model.predict(input_features)[0]  # Predict and extract the result
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
