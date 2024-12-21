"""
Application that predicts laptop prices based on user inputs.

Trained on a dataset containing features like screen size, RAM, weight,
company, type, CPU, memory, GPU, and operating system.

"""

import os
import numpy as np
from flask import Flask, request, render_template
import pickle

# Create an app object using the Flask class
app = Flask(__name__)

# Load the trained model (Pickle file)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

@app.route('/')
def home():
    """Renders the homepage with the input form."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Processes the user input from the form, prepares features for the model,
    makes a prediction, and displays the result on the webpage.
    """
    try:
        # Extract input values from the form
        input_data = request.form.to_dict()

        # Map categorical features to one-hot encoded values
        company_mapping = {'Asus': [1, 0, 0, 0, 0], 'Dell': [0, 1, 0, 0, 0],
                           'HP': [0, 0, 1, 0, 0], 'Lenovo': [0, 0, 0, 1, 0],
                           'Other': [0, 0, 0, 0, 1]}
        type_mapping = {'Gaming': [1, 0, 0, 0, 0], 'Netbook': [0, 1, 0, 0, 0],
                        'Notebook': [0, 0, 1, 0, 0], 'Ultrabook': [0, 0, 0, 1, 0],
                        'Workstation': [0, 0, 0, 0, 1]}
        cpu_mapping = {'High-Performance': [1, 0, 0], 'Mid-Range': [0, 1, 0],
                       'Other': [0, 0, 1]}
        memory_mapping = {'HDD': [1, 0, 0, 0], 'Hybrid': [0, 1, 0, 0],
                          'Mixed': [0, 0, 1, 0], 'SSD': [0, 0, 0, 1]}
        gpu_mapping = {'ARM Mali T860 MP4': [1, 0, 0], 'Intel': [0, 1, 0],
                       'Nvidia': [0, 0, 1]}
        os_mapping = {'Android': [1, 0, 0, 0, 0, 0], 'Chrome OS': [0, 1, 0, 0, 0, 0],
                      'Linux': [0, 0, 1, 0, 0, 0], 'Windows 10': [0, 0, 0, 1, 0, 0],
                      'Windows 10 S': [0, 0, 0, 0, 1, 0], 'Windows 7': [0, 0, 0, 0, 0, 1]}

        # Validate input and prepare numerical and categorical features
        inches = float(input_data.get('inches', 0))
        ram = float(input_data.get('ram', 0))
        weight = float(input_data.get('weight', 0))
        company = company_mapping.get(input_data.get('company', 'Other'), [0, 0, 0, 0, 1])
        type_name = type_mapping.get(input_data.get('type_name', 'Notebook'), [0, 0, 1, 0, 0])
        cpu = cpu_mapping.get(input_data.get('cpu', 'Other'), [0, 0, 1])
        memory = memory_mapping.get(input_data.get('memory', 'HDD'), [1, 0, 0, 0])
        gpu = gpu_mapping.get(input_data.get('gpu', 'Intel'), [0, 1, 0])
        os = os_mapping.get(input_data.get('os', 'Windows 10'), [0, 0, 0, 1, 0, 0])

        # Combine all features into a single array
        features = [inches, ram, weight] + company + type_name + cpu + memory + gpu + os
        features = np.array([features])  # Reshape for model input

        # Make a prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        final_output=58970+(output*37243.2)

        # Render the result back to the webpage
        return render_template('index.html', prediction_text=f'Predicted Laptop Price:{final_output} Euros')
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=False)
