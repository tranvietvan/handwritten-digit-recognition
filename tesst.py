import numpy as np
import pickle
import gzip
from sklearn import svm
from sklearn import metrics
from joblib import dump
import time

def read_mnist(mnist_file):
    """
    Reads MNIST data.

    Parameters
    ----------
    mnist_file : string
        The name of the MNIST file (e.g., 'mnist.pkl.gz').

    Returns
    -------
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) : tuple
        train_X : numpy array, shape (N=50000, d=784)
            Input vectors of the training set.
        train_Y: numpy array, shape (N=50000)
            Outputs of the training set.
        val_X : numpy array, shape (N=10000, d=784)
            Input vectors of the validation set.
        val_Y: numpy array, shape (N=10000)
            Outputs of the validation set.
        test_X : numpy array, shape (N=10000, d=784)
            Input vectors of the test set.
        test_Y: numpy array, shape (N=10000)
            Outputs of the test set.
    """
    f = gzip.open(mnist_file, 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    train_X, train_Y = train_data
    val_X, val_Y = val_data
    test_X, test_Y = test_data

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

train_X, train_Y, val_X, val_Y, test_X, test_Y = read_mnist('mnist.pkl.gz')

#Training test range to choose model
TRAINING_TEST=10000

train_test_X=train_X[:TRAINING_TEST]
train_test_Y=train_Y[:TRAINING_TEST]

#Define C
C=[0.001,0.1,1,10,100]

#testing
x=1
model_linear = svm.SVC(kernel='linear', C=x)

# Traning + measuring time
start = time.process_time()
model_linear.fit(train_test_X, train_test_Y)
process_time = time.process_time() - start

# Test with value
prediction_val = model_linear.predict(val_X)
accurate_val = metrics.accuracy_score(val_Y, prediction_val)
prediction_train = model_linear.predict(train_X)
accurate_train = metrics.accuracy_score(train_Y, prediction_train)

# Print time need to train and accurate
print("accuracy in train with C=", x, ": ", accurate_train)
print("accuracy in validation with C=", x, ": ", accurate_val)
print("time take to process: ", process_time)

predict1=model_linear.predict([test_X[10]])
print("prediction:",predict1)
print("value:",test_Y[10])

#Testing with linear kernel
'''for x in C:
    #Make the model
    model_linear=svm.SVC(kernel='linear',C=x)

    #Traning + measuring time
    start = time.process_time()
    model_linear.fit(train_test_X,train_test_Y)
    process_time = time.process_time() - start

    #Test with value
    prediction_val=model_linear.predict(val_X)
    accurate_val=metrics.accuracy_score(val_Y,prediction_val)
    prediction_train=model_linear.predict(train_X)
    accurate_train=metrics.accuracy_score(train_Y,prediction_train)

    #Print time need to train and accurate
    print("accuracy in train with C=",x,": ",accurate_train)
    print("accuracy in validation with C=",x,": ",accurate_val)
    print("time take to process: ",process_time)
'''