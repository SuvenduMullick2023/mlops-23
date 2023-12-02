from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Function to load a model based on the model type
def load_model(model_type):
    # Assuming you have saved SVM, LR, and Decision Tree models in the models folder
    model_filename = f"models/{model_type}_model.joblib"
    loaded_model = joblib.load(model_filename)
    return loaded_model

# Sample data for prediction (replace this with your actual data structure)
sample_data = np.random.rand(1, 5)

# Predict API route
@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    try:
        # Load the specified model
        model = load_model(model_type)

        # Get the input data from the request
        input_data = request.get_json()

        # Perform prediction
        prediction = model.predict(np.array([input_data]))  # Adjust this based on your data format

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
