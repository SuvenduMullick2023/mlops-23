from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics
# we will put all utils here
# Standard scientific Python imports
import matplotlib.pyplot as plt


def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    #2. Qualitative sanity check of the data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    return X, y 

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5,random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# train the model of choice with the model prameter
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # train the model # Learn the digits on the train subset
    model.fit(x, y)
    return model




def split_train_dev_test1(X, y, test_size, dev_size):
    # Split data into 60% train and 20% test subsets and 20% validation
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False,random_state=1)
    X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_dev, y_train_dev, test_size=dev_size, shuffle=False,random_state=1)
    return X_train,X_test,X_dev,y_train,y_test,y_dev

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )


    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
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
    )

def predict_and_eval1(model, X_test, y_test):
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