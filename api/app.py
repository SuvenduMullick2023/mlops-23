from flask import Flask, request, jsonify
from joblib import dump, load
import numpy as np
from PIL import Image
import io ,os

app = Flask(__name__)

@app.route("/")
def hello_world1():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}

@app.route("/predict", methods = ['POST'])
def pred_image():
    js = request.get_json( )
    image1 = js['image']
    best_model = load('./models/svm_gamma_0.001_C_1_kernel_rbf.joblib')
    image1_1d = np.array(image1).reshape(1, -1)
    predicted1 = best_model.predict(image1_1d)
    print(predicted1)
    return str(predicted1)