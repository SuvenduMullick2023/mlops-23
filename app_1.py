from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

#from exp import best_model_path

app = Flask(__name__)
model = load_model(best_model_path)

@app.route('/compare_images', methods=['POST'])
def compare_images_route():
    try:
        image1 = request.files['image1']
        image2 = request.files['image2']

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
    app.run(debug=True)
