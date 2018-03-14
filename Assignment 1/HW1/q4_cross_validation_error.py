import numpy as np
import math
from q4_train import q4_train
from q4_predict import q4_predict
from q4_mse import q4_mse


def q4_cross_validation_error(X, Y, lambdavec, mode, N):
    # Calculates the cross-validation errors for different values of lambdavec, given the
    # training set X, Y.
    #
    # ** Implementation notes **
    # - As discussed in class, you should first randomly permute the examples, before starting the
    #   cross-validation stage. Here we did it for you: we created Xr and Yr which are obtained from
    #   X and Y by permuting examples. You should use Xr and Yr in your code (not X and Y)
    # - Do not change/initialize/reset the Python pseudo-number generator.
    #
    # INPUT
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input example
    #  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #     i-th element is the correct output value for the i-th input example.
    #  lambdavec: a numpy.ndarray vector of size [k x 1] and type 'float'
    #             containing the set of regularization hyperparameter values
    #  mode: specifies the type of features;
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #  N: `int' representing the number of folds for the cross-validation stage
    #
    # OUTPUT
    #  error: a numpy.ndarray vector of size [k x 1] and type 'float'
    #         containing the cross-validation error (i.e., the average of the mean
    #         squared errors over the N validation sets) for each value in lambdavec.
    #

    # ********  DO NOT TOUCH THE FOLLOWING 5 LINES  ********************
    np.random.seed(0)
    m = X.shape[0]
    idxperm = np.random.permutation(m) - 1
    Xr = X[idxperm, :]
    Yr = Y[idxperm]
    # ******************************************************************

    # insert your code here
    # make sure to use Xr and Yr in your code, NOT X and Y (read Implemantiation notes in the header)

    # splitting the data into N folds
    error = []
    splitX = np.split(Xr, N)
    splitY = np.split(Yr, N)
    # loop through upto the size of lambdavac
    for i in range(0, lambdavec.shape[0]):
        lambd = lambdavec[i]
        totalerrrors = []

        for j in range(0, N):
            # getting the training data
            X_Train = list(splitX)
            Y_Train = list(splitY)

            #gettting the test data
            Y_test = splitY[j]
            X_Test = splitX[j]

            # getting sub-training data
            data_Train = X_Train[0]

            for data in range(0, len(X_Train) - 1):
                data_Train = np.concatenate((data_Train, X_Train[data + 1]), axis=0)

            # Getting the new Y data for training
            y_data = np.array(Y_Train).ravel()

            # Training data to get our theta
            theta = q4_train(data_Train, y_data, lambd, mode)

            # Getting our predicted Y
            predict_Y = q4_predict(theta, X_Test, mode)

            # Calculating the errors and their mean
            allErrors = q4_mse(predict_Y, Y_test)

            # mean  error
            totalerrrors.append(allErrors)
            mean_error = np.mean(totalerrrors)
        error.append(mean_error)
    return error
