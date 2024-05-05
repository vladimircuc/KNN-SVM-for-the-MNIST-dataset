import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import pdb
from sklearn import svm
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description= "Train and test a support vector mechine")
parser.add_argument('--debug', action = 'store_true', help='use pbd')

def plot_matrix(cm, cmap=plt.cm.RdBu_r, title="Confusion Matrix"):
    classes = ["0","1","2","3","4","5","6","7","8","9"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = range(len(classes))
    plt.xticks(marks, classes, rotation=45)
    plt.yticks(marks, classes)
    fmt = 'd'
    thold = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha='center',
                     va='center', color='Black' if cm[i, j] >= thold else 'White')
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def main():
    clf = svm.SVC(kernel='poly', C=3)
    print("Loading data into the variables...")

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    xtrain = xtrain.reshape(60000, 784)
    xtest = xtest.reshape(10000, 784)

    x_train, x_test, y_train, y_test = train_test_split(xtest, ytest, train_size=0.10)
    input("Press <Enter> to train this model...")

    clf.fit(xtrain, ytrain)

    print("Model trained, calculating predictions for the test data...")
    pred = clf.predict(xtest)

    accuracy = np.mean(pred == ytest)
    print(f"Accuracy = {accuracy * 100:0.2f}%")

    cm = confusion_matrix(ytest, pred)
    plot_matrix(cm)




if __name__ == '__main__':
    main()