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
from utils import preprocess_data, read_digits, predict_and_eval, split_train_dev_test ,tune_hparams_svm,tune_hparams_tree,tune_hparams_LR
from joblib import dump, load
from sklearn import tree
import os
from pathlib import Path
import pandas as pd
import argparse
from sklearn import preprocessing


parser = argparse.ArgumentParser(description = "Digit Classification application")
parser.add_argument("--model_type", type= str, default ="svm", help="clf_name",choices=["svm","tree","LR"])
parser.add_argument("--random_state", type= int,default = 42, help="random_state-choose number between 0-1000")
args = parser.parse_args()
#result = args.number * 2
#print("Result:", result)
print("model_type =" ,args.model_type)

print("random_state =", args.random_state)

output_result = []
output_result1 =[]
output_result2 =[]
iteration_no = 1
for iteration in  range(iteration_no) :
# 1. Get the dataset
    X, y = read_digits()
    
    # 3. Data splitting -- to create train and test sets
    
    # X_train, X_test, y_train, y_test = split_data(X,y, test_size=0.3)


    test_size = [0.1, 0.2, 0.3,0.4]
    dev_size = [0.1, 0.2, 0.3,0.4]
    for i in test_size :
        for j in dev_size : 
            X_train, X_test,X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=i, dev_size=j,random_state = args.random_state)
            
            # 4. Data preprocessing
            #X_train = preprocessing.normalize([X_train])
            #X_dev = preprocessing.normalize([X_dev])
            #X_test = preprocessing.normalize([X_test])

            X_train = preprocess_data(X_train)
            X_dev = preprocess_data(X_dev)
            X_test = preprocess_data(X_test)
            
            
            
            
            
            
            
            # List of solver options to iterate through
            solver_options = ['saga', 'newton-cg', 'liblinear', 'sag', 'saga','newton-cg','lbfgs']
            param_dict_LR ={}
            param_dict_LR['solver'] = solver_options
            param_dict_LR['random_state'] = [int(args.random_state)]

            # 5. Model training
            gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
            C_list = [0.1, 1, 10, 100, 1000]
            param_dict_svm ={}
            param_dict_svm = {'gamma': gamma_list,'C' : C_list,'kernel' :['rbf','linear']}
            
            max_depth_list = [5, 10, 15, 20, 50, 100]
            param_dict_tree = {}
            param_dict_tree['max_depth'] = max_depth_list
            param_dict_tree['random_state'] = [int(args.random_state)]
            model_list = [args.model_type]
            #model_list = ['svm', 'tree']
            for model_type in model_list:
                if model_type =="tree" :
                    best_hparams, best_model_path, best_accuracy = tune_hparams_tree( X_train, y_train,X_dev, y_dev,param_dict_tree,model_type ='tree')
                    
                    # loading of model         
                    best_model = load(best_model_path)
                    
                    train_acc,_ = predict_and_eval(best_model, X_train, y_train)
                    dev_acc,_ = predict_and_eval(best_model, X_dev, y_dev)
                    test_acc,_ = predict_and_eval(best_model, X_test, y_test)
                    
                    current_results = {'model_type': model_type, 'iteration': iteration, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                    output_result1.append(current_results)
                    #print(metrics.confusion_matrix(y_test, predicted_y))
                    print("Accuracy_test  : {0:2f}% Accuracy_dev  : {1:2f}% Accuracy_train  : {2:2f}% ".format((test_acc*100),(dev_acc*100),(train_acc*100)))
                    # Create a DataFrame to store results
                    df_output_result = pd.DataFrame(output_result1 )
                    
                    #my_path = os.path.abspath(os.path.dirname(__file__))
                    # Create a folder to store results if it doesn't exist
                    #results_folder = "results"
                    #os.makedirs("../results", exist_ok=True)

                    results_csv_filename = f"results_{args.model_type}_{args.random_state}.csv"
                    # Save the DataFrame to a CSV file
                    my_path = os.path.abspath(os.path.dirname(__file__))
                    path = os.path.join(my_path, "results/"+ results_csv_filename)
                    
                    df_output_result.to_csv(path, index=False)
                    # loading of model         
                    #best_model = load(best_model_path)
                if model_type == "svm" :
                    best_hparams, best_model_path, best_accuracy = tune_hparams_svm( X_train, y_train,X_dev, y_dev,param_dict_svm,model_type ='svm')
                #best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
                #y_dev, h_params_combinations)
                    
                    # loading of model         
                    best_model = load(best_model_path)
                    
                    train_acc,_ = predict_and_eval(best_model, X_train, y_train)
                    dev_acc,_ = predict_and_eval(best_model, X_dev, y_dev)
                    test_acc,_ = predict_and_eval(best_model, X_test, y_test)
                    current_results = {'model_type': model_type, 'iteration': iteration, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                    output_result2.append(current_results)
                    
                    print("Accuracy_test  : {0:2f}% Accuracy_dev  : {1:2f}% Accuracy_train  : {2:2f}% ".format((test_acc*100),(dev_acc*100),(train_acc*100)))

                    # Create a DataFrame to store results
                    df_output_result = pd.DataFrame(output_result2 )
                    
                    #my_path = os.path.abspath(os.path.dirname(__file__))
                    # Create a folder to store results if it doesn't exist
                    #results_folder = "results"
                    #os.makedirs("../results", exist_ok=True)

                    results_csv_filename = f"results_{args.model_type}_{args.random_state}.csv"
                    # Save the DataFrame to a CSV file
                    my_path = os.path.abspath(os.path.dirname(__file__))
                    path = os.path.join(my_path, "results/"+ results_csv_filename)
                    
                    df_output_result.to_csv(path, index=False)
                    # loading of model         
                    #best_model = load(best_model_path)
                if model_type == "LR" :
                    best_hparams, best_model_path, best_accuracy = tune_hparams_LR( X_train, y_train,X_dev, y_dev,param_dict_LR,model_type ='LR')
                #best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
                #y_dev, h_params_combinations)
                    
                    # loading of model         
                    best_model = load(best_model_path)
                    
                    train_acc,_ = predict_and_eval(best_model, X_train, y_train)
                    dev_acc,_ = predict_and_eval(best_model, X_dev, y_dev)
                    test_acc,_ = predict_and_eval(best_model, X_test, y_test)
                    current_results = {'model_type': model_type, 'iteration': iteration, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                    output_result2.append(current_results)
                    
                    print("Accuracy_test  : {0:2f}% Accuracy_dev  : {1:2f}% Accuracy_train  : {2:2f}% ".format((test_acc*100),(dev_acc*100),(train_acc*100)))

                    # Create a DataFrame to store results
                    df_output_result = pd.DataFrame(output_result2 )
                    
                    #my_path = os.path.abspath(os.path.dirname(__file__))
                    # Create a folder to store results if it doesn't exist
                    #results_folder = "results"
                    #os.makedirs("../results", exist_ok=True)

                    results_csv_filename = f"results_{args.model_type}_{args.random_state}.csv"
                    # Save the DataFrame to a CSV file
                    my_path = os.path.abspath(os.path.dirname(__file__))
                    path = os.path.join(my_path, "results/"+ results_csv_filename)
                    
                    df_output_result.to_csv(path, index=False)




                print('Train : {0} Test_size : {1} Dev_size :{2}'.format( round(1-i-j,1) , i,  j)," ", end = "")
#print(pd.DataFrame(output_result).groupby('model_type').describe().T)
                # 6. Getting model predictions on test set
                # 7. Qualitative sanity check of the predictions
                # 8. Evaluation

                
        
