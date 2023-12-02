from flask import Flask, request, jsonify
import joblib
import numpy as np
from joblib import dump, load

app = Flask(__name__)

# Function to load a model based on the model type
'''def load_model(model_type):
    # Assuming you have saved SVM, LR, and Decision Tree models in the models folder
    model_filename = f"models/{model_type}_model.joblib"
    loaded_model = joblib.load(model_filename)
    return loaded_model

# Sample data for prediction (replace this with your actual data structure)
sample_data = np.random.rand(1, 5)'''

# Predict API route
@app.route('/predict/<model_type>', methods=['POST'])


def pred_image(model_type):
    js = request.get_json( )
    image1 = js['image']
    if model_type == "svm":
        best_model = load('./models/svm_m22aie218gamma_0.0001_C_10_kernel_linear.joblib')
        image1_1d = np.array(image1).reshape(1, -1)
        predicted1 = best_model.predict(image1_1d)
        print(predicted1)
        return str(predicted1)
    elif model_type == "tree":
        best_model = load('./models/tree_max_depth_15_random_state_42.joblib')
        image1_1d = np.array(image1).reshape(1, -1)
        predicted1 = best_model.predict(image1_1d)
        print(predicted1)
        return str(predicted1)
    elif model_type == "LR":
        best_model = load('./models/LR_m22aie218solver_liblinear_random_state_42.joblib')
        image1_1d = np.array(image1).reshape(1, -1)
        predicted1 = best_model.predict(image1_1d)
        print(predicted1)
        return str(predicted1)
    



if __name__ == '__main__':
    app.run(debug=True)
