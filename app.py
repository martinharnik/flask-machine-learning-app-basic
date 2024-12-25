import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# Create an instance of the Flask class
app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Define the route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get the data from the POST request
    data = request.json['data']
    print(data)
    # Convert the data into a numpy array
    print(np.array(list(data.values())).reshape(1, -1))
    # Scale the data
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    # Make the prediction
    output = regmodel.predict(new_data)
    print(output[0])
    # Return the prediction
    return jsonify(output[0])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

