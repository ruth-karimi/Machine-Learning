import numpy as np


def q4_features(X, mode):

    # Given the data matrix X (where each row X[i,:] is an example), the function computes the feature matrix B,
    #  where row B[i,:] represents the feature vector
    # associated to example X[i,:]. The features should be either linear or quadratic
    # functions of the inputs, depending on the value of the input argument 'mode'.
    # Please make sure to implement the features according to the *exact* order
    # specified in the text of the homework assignment.

    # INPUT:
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #  is a d-dimensional input example
    #  mode: specifies the type of features
    #  it is a 'str' that can be either 'linear' or 'quadratic'.

    # OUTPUT:
    #  B: a numpy.ndarray matrix of size [m x n] and type 'float', with each row
    #  containing the feature vector of an example

    if mode == 'linear':
        # inserting ones in B; adding one column
        B = np.insert(X, 0, 1, axis=1)

    elif mode == 'quadratic':
        array1 = []

        # get m from X and extend it to data from index 1 of row
        for i, m in enumerate(X):
            row = [1]
            row.extend(m)

            # looping to the length of m
            for j in range(len(m)):
                for u in range(len(m)):

                    # append to create arrays of final answers
                    row.append(m[u] * m[j])

            # append for final array
            array1.append(row)
        # Make B an array
        B = np.array(array1)

    else:
        print('Error, only linear and quadratic forms are supported');
        return []

    return B
