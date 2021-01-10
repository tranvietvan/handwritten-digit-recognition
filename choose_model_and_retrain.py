from sklearn import svm
from sklearn import metrics
from joblib import dump
import pickle
import gzip
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

#make the final model
final_model=svm.SVC(kernel='rbf',C=10,gamma=0.001)

#train the model with full train data
start = time.process_time()
final_model.fit(train_X,train_Y)
process_time = time.process_time() - start

#save the model for live predict
dump(final_model, "model/final_model")

#predict in test value and calculate mean
prediction=final_model.predict(test_X)
accurate=metrics.accuracy_score(test_Y,prediction)

print("accuracy:",accurate)
print("time take to process:",process_time)



