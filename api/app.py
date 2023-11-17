from flask import Flask

app = Flask(__name__)
val =1
@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

from flask import Flask, request, jsonify
from joblib import dump, load
import numpy as np
from PIL import Image
import io ,os


app = Flask(__name__)

@app.route("/")
def hello_world():
    return '<h1>Hello from Flask & Docker</h2>'
    

#best_model_path = "/home/suvendu/digit_classification/mlops-23/models"

best_model_path = "models/svm_gamma_0.0001_C_100_kernel_rbf.joblib"
#files=os.listdir(os.path.join(best_model_path))
#print(files)

#path = best_model_path + os.sep+files[0]

model = load(best_model_path)

@app.route("/predict", methods=["POST"])


def predict():
    try:
        js = request.get_json( )
        print("ok")
        #image1 = request.files['image1']
        #image2 = request.files['image2']
        image1 = js['image1']
        

        print("image1")
        image2 = js['image2']
        
        print("image2")

        image1 = Image.open(io.BytesIO(image1.read())).convert('L').resize((28, 28))
        image2 = Image.open(io.BytesIO(image2.read())).convert('L').resize((28, 28))

        image1 = np.array(image1) / 255.0
        image2 = np.array(image2) / 255.0

        image1 = image1.reshape(1, 28, 28, 1)
        image2 = image2.reshape(1, 28, 28, 1)

        prediction1 = np.argmax(model.predict(image1))
        prediction2 = np.argmax(model.predict(image2))

        result = (prediction1 == prediction2)

        return jsonify({'result': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        


if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000,debug=True)

       