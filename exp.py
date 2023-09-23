"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""
 # Assignment 2 and 3 is updated
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from utils import preprocess_data, read_digits, predict_and_eval, split_train_dev_test ,tune_hparams

# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train and test sets

# X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)
test_size = [0.1, 0.2, 0.3,0.4]
dev_size = [0.1, 0.2, 0.3,0.4]
for i in test_size :
    for j in dev_size : 
        X_train, X_test,X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=i, dev_size=j)
        
        # 4. Data preprocessing
        X_train = preprocess_data(X_train)
        X_dev = preprocess_data(X_dev)
        X_test = preprocess_data(X_test)
        print('Train : {0} Test_size : {1} Dev_size :{2}'.format( len(X_train) , len(X_dev),len(X_test)))
        print("Total sample :",(len(X_train)+ len(X_dev)+len(X_test)) )

        
        
        
        
        
        
        
        # 5. Model training
        param_dict = {'gamma': [0.001,0.01,0.1,1],'C' : [1.0,10.0,20.0],'kernels' :['rbf','linear']}
        
        model = tune_hparams( X_train, y_train,X_dev, y_dev,param_dict)

        print('Train : {0} Test_size : {1} Dev_size :{2}'.format( round(1-i-j,1) , i,  j)," ", end = "")

        # 6. Getting model predictions on test set
        # 7. Qualitative sanity check of the predictions
        # 8. Evaluation
        predict_and_eval(model, X_test, y_test)
