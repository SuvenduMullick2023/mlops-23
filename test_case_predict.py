from  api.app import app
import pytest
from sklearn import datasets
import pdb
import pytest
import json


               




def label_digit(image_label, one_d_list):
    response = app.test_client().post("/predict", json={"image": one_d_list})
    assert response.status_code == 200
    response_data = (response.get_data(as_text=True))
    predicted_digit = int(response_data.strip('[]'))
    assert image_label == predicted_digit


lst=[0, 0, 0 , 0 , 0, 0, 0, 0 , 0 , 0]
def test_post_predict():
    image_label = 1
    response = app.test_client().post("/predict", json={"image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]})
    assert response.status_code == 200
    response_data = (response.get_data(as_text=True))
    predicted_digit = int(response_data.strip('[]'))
    assert image_label == predicted_digit


    digits = datasets.load_digits();
    X = digits.images
    y = digits.target
    
    
    count = 0
    for j in range(0, len(y)):
        if lst[y[j]] == 0:
            lst[y[j]] = 1
            print("label:", y[j])
            x = [element for row in X[j] for element in row]    
            label_digit(y[j], x)
            count = count + 1
        if (count == 10):
            break
        j = j + 1
                      
