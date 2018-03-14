import numpy as np
from q4_train import q4_train
from q4_predict import q4_predict
from q4_mse import q4_mse


def q4_test_error(X, Y, Xtest, Ytest, lambdavec, mode):

    # Given training and test set, it trains the model and calculates the test error.
    #
    # INPUT
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input training example
    #  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #     i-th element is the correct output value for the i-th input training example.
    #  Xtest: a numpy.ndarray vector of size [M x d] and type 'float', where
    #         each row is a d-dimensional test example
    #  Ytest: a numpy.ndarray vector of size [M x 1] and type 'float',
    #         containing the output values of the test examples
    #  lambdavec: a numpy.ndarray vector of size [k x 1] and type 'float'
    #             containing the set of regularization hyperparameter values
    #  mode: specifies the type of features;q
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #
    # OUTPUT
    #  error: a numpy.ndarray vector of size [k x 1] and type 'float'
    #         containing the test errors, one for each value in lambdavec.
    #

    # insert your code here
    error = []

    for i in range(0, lambdavec.shape[0]):
        # creating an array that holds total errors when lambdavaec is at i
        total_error = []

        # getting the theta from training
        # training data given X, Y, Lambdavec at position i and  mode
        theta = q4_train(X, Y, lambdavec[i], mode)

        # getting the predicted Y from the predict function
        predict_Y = q4_predict(theta, Xtest, mode)

        # getting error from mse function
        one_error = q4_mse(predict_Y, Ytest)

        total_error.append(one_error)

        # getting the mean of the total errors and appending it to array error
        mean_error = np.mean(total_error)
        error.append(mean_error)

    return error
