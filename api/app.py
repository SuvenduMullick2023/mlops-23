from flask import Flask, request, jsonify
from joblib import dump, load
import numpy as np
from PIL import Image
import io ,os



@app.route("/predict", methods = ['POST'])
def pred_image():
    js = request.get_json( )
    image1 = js['image']
    best_model = load('./models/svm_gamma:0.001_C:1.joblib')
    image1_1d = np.array(image1).reshape(1, -1)
    predicted1 = best_model.predict(image1_1d)
    print(predicted1)
    return str(predicted1)