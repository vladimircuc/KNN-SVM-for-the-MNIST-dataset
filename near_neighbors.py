import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pdb 
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix




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
    #seed = 4
    k = 3 #nr of neighors
    print("Loading data into the variables...")

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    xtrain = xtrain.reshape(60000, 784)
    xtest = xtest.reshape(10000, 784)

    # Train the knn model 
    
    input("Press <Enter> to train this model...")
    
    clf = KNeighborsClassifier(k)
    clf.fit(xtrain, ytrain,)

    print("Model trained, calculating predictions for the test data...")
    pred = clf.predict(xtest)

    accuracy = np.mean(pred == ytest)

    print(f"Accuracy = {accuracy * 100:0.2f}%")

    cm = confusion_matrix(ytest, pred)
    plot_matrix(cm)


    #Show results

    #xm, ym = np.meshgrid(np.arange(-0.1, 1.1, 0.002), np.arange(-0.1, 1.1, 0.002))

    #pred = clf.predict(np.c_[xm.ravel(), ym.ravel()]).reshape(xm.shape)
    
    #plt.pcolormesh
    '''
    pred = clf.predict(xtest)

    accuracy = np.mean(pred == ytest)
    print(accuracy*100)

    


    # Assuming cm is the confusion matrix and classes is a list of class labels
    # Initialize arrays to store precision and recall for each class
    pre = np.zeros(10)
    rec = np.zeros(10)
    for i in range(10):
    # Precision calculation
        pre[i] = cm[i, i] / np.sum(cm[:, i])  # Sum along the column
        # Recall calculation
        rec[i] = cm[i, i] / np.sum(cm[i, :])  # Sum along the row

        
    print(pre)
    print(rec)
    '''


    #x_test = np.random.rand(10000, numfeatures)
    #y_test = clf.predict(x_test)


if __name__ == '__main__':
    main()