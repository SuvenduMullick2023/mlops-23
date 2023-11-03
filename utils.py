from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics 
from sklearn.model_selection import GridSearchCV
import numpy as np
import itertools  
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn import tree
import pdb, os
#from skimage.transform import rescale, resize, downscale_local_mean
#import  matplotlib as plt
# we will put all utils here





def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    #print("Image Data Shape" , digits.data[0].shape)
    return X, y 

def resize_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    '''for row in range(1797):
        for i in range(0,63):
        newfeatures.iloc[row,16*i]=features[row][i]
        newfeatures.fillna(method="ffill",axis=1,inplace=True)'''
def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data



def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into 60% train and 20% test subsets and 20% validation
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    print( 'train_dev_size =', len(X_train_dev))
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=dev_size/(1-test_size), shuffle=False)
    print('train_size =',len(X_train))
    return X_train,X_test,X_dev,y_train,y_test,y_dev



# # 6. Getting model predictions on test set
# 7. Qualitative sanity check of the predictions
# 8. Evaluation
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    '''print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )'''
    accuracy_test = accuracy_score(predicted, y_test)
    f1_value = metrics.f1_score(y_test, predicted, average="macro")
    return accuracy_test ,f1_value
    
    #print("---------------------------------------------------------------")
    '''disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")


    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )'''

def hparams_combination(param_dict) :
    keys, values = zip(*param_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(permutations_dicts)
    return permutations_dicts
def tune_hparams_svm( X_train, Y_train,x_dev, y_dev,list_of_all_param_combination,model_type="svm"):
    
    keys, values = zip(*list_of_all_param_combination.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    #print("The combinations dictionary : " + str(permutations_dicts))
    
    
    # default setting     
    g = 'scale'
    cval = 1.0
    kval = 'rbf'
    best_score =[0,0]
    best_model = None
    avg_scores ={}
    best_hparams = {}
    for model_params in permutations_dicts :
        #print(dict1)
        '''for k,v in dict1.items():
                    
                    if k == 'gamma':
                        g = v
                    elif k =='C':
                        cval = v
                    elif k == 'kernels':
                        kval = v'''
        #print(kval,cval,g)    
        #cur_model =svm.SVC(kernel =kval,C=cval,gamma= g)
        #cur_model =svm.SVC(**model_params)
        cur_model = train_model(X_train, Y_train, model_params, model_type=model_type)
        
        cv_scores, _  = predict_and_eval(cur_model, x_dev, y_dev)
        avg_scores[str(kval) + "-" + str(cval)+ "-" + str(g)]= round(np.average(cv_scores),3)
        best_model_path = ""
        if best_score[1] < round(np.average(cv_scores),3) :
            best_model = cur_model
            best_score[1] = round(np.average(cv_scores),3)
            best_score[0] = str(kval) + "-" + str(cval)+ "-" + str(g)
            best_hparams = model_params
            y_pred_train =best_model.predict(X_train)
            best_accuracy = best_score[1]
            #accuracy_train, f1_value_train  = predict_and_eval(best_model, X_train, Y_train)
            #accuracy_dev, f1_value_dev  = predict_and_eval(best_model, x_dev, y_dev)
            
            #accuracy_train = accuracy_score(y_pred_train, Y_train)
            #y_pred_dev =best_model.predict(x_dev)
            #accuracy_dev = accuracy_score(y_pred_dev, y_dev)
    
    os.makedirs('models', exist_ok=True)
        
    model_path = "./models/{}_".format(model_type) +"_".join(["{}_{}".format(k,v) for k,v in best_hparams.items()]) + ".joblib"
    print(model_path)
    # save the best_model    
    dump(best_model, model_path)
    #print('All_scores :' ,avg_scores)
    #print("Optimal parameters ::: Kernel--C--gamma  : ", best_score)
    #print(" SVM == Accuracy Train : {0:2f}%  Accuracy_Dev :{1:2f}%".format(accuracy_train*100, accuracy_dev*100 ))
    return best_hparams, model_path, best_accuracy

# train the model of choice with the model prameter    
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
        model = clf(**model_params)
    # train the model
        model.fit(x, y)
    if model_type == "tree":
        # Create a classifier: a support vector classifier
        clf = tree.DecisionTreeClassifier
        #print(model_params)
        model = clf(**model_params)
    # train the model
        model.fit(x, y)
    return model    


def tune_hparams_tree( X_train, Y_train,x_dev, y_dev,list_of_all_param_combination,model_type ='tree'):
    
    keys, values = zip(*list_of_all_param_combination.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    #print("The combinations dictionary : " + str(permutations_dicts))
    
    
    # default setting     
    best_hparams = {}
    best_score =[0,0]
    best_model = None
    avg_scores ={}
    for model_params in permutations_dicts :
        #print(dict1)
        '''for k,v in dict1.items():
                    
                    if k == 'gamma':
                        g = v
                    elif k =='C':
                        cval = v
                    elif k == 'kernels':
                        kval = v'''
        #print(kval,cval,g)    
        #cur_model =svm.SVC(kernel =kval,C=cval,gamma= g)
        cur_model = train_model(X_train, Y_train, model_params, model_type="tree")
        #cur_model = tree.DecisionTreeClassifier(random_state=0)
        #tree.plot_tree(cur_model)            
    
        #cur_model =svm.SVC(dict1)
        #cur_model.fit(X_train, Y_train)
        #cv_scores = cur_model.score(x_dev, y_dev)
        cv_scores = predict_and_eval(cur_model, x_dev, y_dev)
        #avg_scores[str(kval) + "-" + str(cval)+ "-" + str(g)]= round(np.average(cv_scores),3)
        
        if best_score[1] < round(np.average(cv_scores),3) :
            best_model = cur_model
            best_score[1] = round(np.average(cv_scores),3)
            #best_score[0] = str(kval) + "-" + str(cval)+ "-" + str(g)
            best_hparams = model_params
            best_accuracy = best_score[1]
            #y_pred_train = best_model.predict(X_train)
            #accuracy_train = accuracy_score(y_pred_train, Y_train)
            #y_pred_dev = best_model.predict(x_dev)
            #accuracy_dev = accuracy_score(y_pred_dev, y_dev)
    
    model_path = "./models/{}_".format(model_type) +"_".join(["{}_{}".format(k,v) for k,v in best_hparams.items()]) + ".joblib"
    print(model_path)
    # save the best_model    
    dump(best_model, model_path)
    #print('All_scores :' ,avg_scores)
    #print("Optimal parameters ::: Kernel--C--gamma  : ", best_score)
    #print(" Tree   == Accuracy Train : {0:2f}%  Accuracy_Dev :{1:2f}%".format(accuracy_train*100, accuracy_dev*100 ))
    return best_hparams, model_path, best_accuracy    