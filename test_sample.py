# content of test_sample.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

from utils import hparams_combination , split_train_dev_test ,read_digits
'''def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 5

def test_wrong_answar():
    assert not inc(3)== 5'''  

def test_hparam_combination_count():
    # test all possible param combinations 
    gama_list = [0.001,0.01,0.1,1]
    C_list = [1.0,10.0,20.0]
    k_list =['rbf','linear']
    h_param = {}

    h_param['gamma'] = gama_list
    h_param['cList'] = C_list
    h_param['kernels'] =k_list
    h_param_combination = hparams_combination (h_param)
    print(len(h_param_combination))
    assert  len(h_param_combination) == len(gama_list) * len(C_list)*len(k_list)


def test_hparam_combination_check():
    # test all possible param combinations 
    gama_list = [0.001,0.01,0.1,1]
    C_list = [1.0,10.0,20.0]
    k_list =['rbf','linear']

    h_param = {}

    h_param['gamma'] = gama_list
    h_param['cList'] = C_list
    h_param['kernels'] =k_list
    h_param_combination = hparams_combination (h_param)
    print(h_param_combination)
    h_param_comb1 = {'gamma':0.001,'cList': 20.0,'kernels':'linear'}
    h_param_comb2 = {'gamma':0.01,'cList': 1.0}
    assert (h_param_comb1 in h_param_combination  )

'''def test_data_splitting():
    X,y = read_digits()

    X = X [:100,:,:]
    y = y[:100]

    test_size =0.1
    dev_size =0.6
    train_size = 1- test_size -dev_size

    X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test(X, y, test_size, dev_size)

    print("test = {} train = {} dev =  {}".format(len(X_test), len(X_train),len(X_dev)))
    assert (len(X_test) == 10 )
    assert (len(X_train) == 30 )
    assert (len(X_dev) == 60 )'''
    

def test_LR_check():
    

    # Sample data (replace this with your own data loading)
    X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of solver options to iterate through
    solver_options = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    # Your roll number (replace with your actual roll number)
    roll_number = 'm22aie218'

    # Iterate through different solvers
    for solver in solver_options:
        # Create and train the logistic regression model
        model = LogisticRegression(solver=solver)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Report performance metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1_score_macro = metrics.f1_score(y_test, y_pred, average='macro')

        print(f"Solver: {solver}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_score_macro:.4f}")

        # Save the model
        model_filename = f"{roll_number}_lr_{solver}.joblib"
        joblib.dump(model, model_filename)
        print(f"Model saved as: {model_filename}")

        # Test case: Check that the loaded model is a Logistic Regression model
        loaded_model = joblib.load(model_filename)
        assert isinstance(loaded_model, LogisticRegression), f"Error: {model_filename} is not a Logistic Regression model"

        # Test case: Check that the solver name matches (converted to lowercase)
        loaded_solver_name = model_filename.split('_')[2]
        print(loaded_solver_name)
        print(solver.lower())
        assert loaded_solver_name == solver.lower()+".joblib", f"Error: Solver name mismatch in {model_filename}\n"

    print("All test cases passed successfully.")
