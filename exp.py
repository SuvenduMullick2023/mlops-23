"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into 60% train and 20% test subsets and 20% validation
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False)
    X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=dev_size, shuffle=False)
    return X_train,X_test,X_dev,y_train,y_test,y_dev

def predict_and_eval(model, X_test, y_test):
    predicted=model.predict(X_test)

    # 7. Qualitative sanity check of the prediction
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    # 8. Evaluation
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

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

    ###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit val
# ues and the predicted digit values.
    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

# 1. get the data set 
digits = datasets.load_digits()

#2. Qualitative sanity check of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################


# 3. Data Pre processing 
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X = data
y = digits.target
#4. Data spliting to create traing and test set 
# Split data into 60% train and 20% test subsets and 20% validation
test_size ,dev_size = 0.2 ,0.2


X_train, X_test,X_dev, y_train, y_test,y_dev = split_train_dev_test(X,y,test_size ,dev_size)
 


# 5. Model Training 
# Create a classifier: a support vector classifier
model = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
model.fit(X_train, y_train)


#6. getting the model prediction on the test set
 
# Predict the value of the digit on the test subset
predict_and_eval(model,X_test,y_test)













