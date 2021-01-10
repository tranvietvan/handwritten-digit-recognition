import pickle
import gzip
from joblib import load


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

#print first 10 value of test Y
print(test_Y[:10])

#live predict

#load the model
model=load("model/final_model")
#predict a value
predict1=model.predict([test_X[6]])
print("prediction:",predict1)
print("actual value:",test_Y[6])