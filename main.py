import random
import sys
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# hardcode conversion for categorical values to numerical values for diamond characteristics
category_to_dummy_coding = {
    # cut: the bigger, the better cut, higher price
    "cut": {
        "Ideal": 1,
        "Fair": 2,
        "Very Good": 3,
        "Good": 4,
        "Premium": 5
    },
    # color: the bigger numerical value, more yellow, arbitrary to relationship with price
    "color": {
        "D": 1,
        "E": 2,
        "F": 3,
        "G": 4,
        "H": 5,
        "I": 6,
        "J": 7,
    },
    # clarity: the bigger numerical value, the worse, lower price
    "clarity": {
        "IF": 1,
        "VVS1": 2,
        "VVS2": 3,
        "VS1": 4,
        "VS2": 5,
        "SI1": 6,
        "SI2": 7,
        "I1": 8
    }
}


def isfloat(s: str):
    """
    Determine if a string is a float number.
    :param s: the string
    :return: true if the string is a float number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_data(label: str, value: str) -> float:
    """
    Clean the data using dummy coding to transform the string categorical value to a numerical value
    :param label: the feature name
    :param value: the categorical value
    :return: the corresponding numerical value
    """
    return category_to_dummy_coding[label][value]


def split_data(file_name: str, test_percent: int) -> None:
    """
    Split the data file into training and test sets based on the test set percentage randomly.
    :param file_name: the data file name
    :param test_percent: the test set percentage
    :return: none
    """
    # set a seed for the random generator and still get approximately the same result everytime
    seed = 1234
    random.seed(seed)

    # get the file prefix used to name training and test file
    file_prefix = file_name.split('.')[0]
    file = open(file_name)
    train_file = open(file_prefix + " train set.csv", 'w')
    test_file = open(file_prefix + " test set.csv", 'w')

    train_set = []
    test_set = []

    counter = 0
    header_processed = False

    while True:
        line = file.readline()
        # if reach EOF, terminate
        if not line:
            break
        # get a random number between 1 and 100, inclusive and determine if the value is below or above the
        # passed in test set percentage, if not greater, append to test set, otherwise, training set.
        rand_num = random.randint(1, 101)
        # append the header row to both set
        if not header_processed:
            train_set.append(line)
            test_set.append(line)
            header_processed = True
            continue

        if rand_num <= test_percent:
            test_set.append(line)
        else:
            train_set.append(line)
        counter += 1

    # write the content to training and test files
    train_file.writelines(train_set)
    test_file.writelines(test_set)

    # subtract one for header row
    test_count = len(test_set) - 1
    train_count = len(train_set) - 1
    print(f"using seed {seed}")
    print(f"Test set: {round(test_count / counter * 100, 2)}%, number of samples: {test_count}.")
    print(f"Training set: {round(train_count / counter * 100, 2)}%, number of samples: {train_count}.")
    print(f"Test set file name: {test_file.name}.")
    print(f"Training set file name: {train_file.name}.")

    # close file streams
    file.close()
    train_file.close()
    test_file.close()


def parse_data(file_name: str) -> tuple[list[str], list[list[float]]]:
    """
    Parse the data in the file into a matrix of numerical value.
    :param file_name: the data file
    :return: a tuple where the first element is a list of feature names and the second element is a matrix where each
    row represent the value of that feature and the y value
    """
    file = open(file_name, 'r')
    header_line = file.readline()
    header_line = header_line.strip()

    # the first row is the feature names
    features = header_line.split(",")
    n = len(features)
    data = []
    while True:
        line = file.readline()
        if not line:
            break
        values = line.split(",")
        cur_data = []
        for i in range(n):
            # if the value is not a categorical value, just save it as it is, otherwise, use dummy coding
            # to transform it to the pre-defined value
            value = clean_data(features[i], values[i]) if features[i] in category_to_dummy_coding else float(values[i])
            cur_data.append(value)
        data.append(cur_data)
    # close the file after processing
    file.close()
    return features, data


def plot_simple_linear_regression(features: list[str], all_data: list[list[float]]) -> None:
    """
    Plot the simple linear regression using the matplot library using the passed-in data frame
    :param features: a list of feature names
    :param all_data: a data frame containing all the X data and corresponding y values
    :return: none
    """
    n = len(features)
    # y data is the last column of the data
    y = all_data[n - 1]

    for i in range(n - 1):
        X = all_data[i]
        # graph title is y value vs the current X feature name
        current_graph_title = features[n - 1] + " vs " + features[i]
        plt.scatter(X, y, label=current_graph_title, color='b')

        # get the min and max y and X values to set to the number of ticks per axis for better display
        min_y, max_y = min(y), max(y)
        plt.yticks(np.linspace(min_y, max_y, 10))
        min_X, max_X = min(X), max(X)
        plt.xticks(np.linspace(min_X, max_X, 5))
        plt.ylim(min_y, max_y)
        plt.xlim(min_X, max_X)

        # use linear regression to get the model and its properties and fit the line
        model, r_sq = simple_linear_regression(X, y)
        print(model.coef_, model.intercept_, r_sq)
        y_pred = model.predict(np.array(X).reshape((-1, 1)))
        linear_fit_label = f"y={round(model.coef_[0], 2)}x{round(model.intercept_, 2):+}"

        # plot the graphs and save the figure to the current directory
        plt.plot(X, y_pred, label=linear_fit_label, color='r')
        plt.title(current_graph_title)
        plt.xlabel(features[i])
        plt.ylabel(features[n - 1])
        plt.subplots_adjust(left=0.18)
        plt.legend()
        plt.savefig(current_graph_title + ".png")
        plt.show()


def multiple_linear_regression(multi_X_data: list[list[float]], y_data: list[float]) -> tuple[object, Any]:
    """
    Perform the multiple linear regression using scikit learn library based on the passed in multiple X data and y
    values
    :param multi_X_data: X value data frame where each row contains all the X values for a single feature
    :param y_data: y values of the data frame
    :return: the multiple linear model and its r-square coefficient as a tuple
    """
    # turn both input into ndarray used by numpy library
    X = np.array(multi_X_data)
    y = np.array(y_data)

    # perform linear regression and get the model result
    lr = LinearRegression().fit(X, y)
    r_sq = lr.score(X, y)
    print(lr.coef_, lr.intercept_, r_sq)
    return lr, r_sq


def simple_linear_regression(X_data: list[float], y_data: list[float]) -> tuple[object, Any]:
    """
    Perform the simple linear regression using scikit learn library based on the passed in X data and its corresponding
    y values.
    :param X_data: all X values for a single feature
    :param y_data: corresponding y values
    :return: the simple linear model and its r-square coefficient as a tuple
    """
    # turn both input into ndarray used by numpy library, X needs reshape as it is a 1-dimensional array
    X = np.array(X_data).reshape((-1, 1))
    y = np.array(y_data)

    # perform linear regression and get the model result
    lr = LinearRegression().fit(X, y)
    r_sq = lr.score(X, y)
    return lr, r_sq


def polynomial_regression(X_data: list[list[float]], y_data: list[float]) -> list[float]:
    """
    Perform polynomial regression based on the X data and y values.
    :param X_data: n-th degree polynomial X data
    :param y_data: corresponding y values
    :return: a list of coefficient for increasing degrees of polynomial, starting from degree of 0
    """
    # all 3 methods below work the same, these were just some trialing I have done when I did
    # the homework, so I commented them out but still leave them here for learning purpose.

    # Method 1:
    # X = np.linalg.pinv(np.array(X_data))
    # y = np.array(y_data)
    # coefficient = np.matmul(X, y)

    # Method 2:
    # X_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
    # model = LinearRegression().fit(X_, y)
    # print(model.coef_)
    # print(model.intercept_)
    # print(model.score(X_, y))

    # Method 3:
    coefficient = np.linalg.lstsq(np.array(X_data), np.array(y_data), rcond=None)
    print(coefficient[0])
    return coefficient[0]


def compute_3rd_degree_X(X_data: list[float]) -> list[list[float]]:
    """
    Compute up to the 3rd degree values of X data starting from degree of 0.
    :param X_data: X data of a single feature
    :return: the computed 0 to 3rd degree polynomial X data.
    """
    # make X to the power from 0 to 3 and save the data
    res = []
    for X in X_data:
        cur = []
        for i in range(4):
            cur.append(X ** i)
        res.append(cur)
    return res


def plot_3rd_degree_polynomial(coefficients: list[float], X_data: list[float], y_data: list[float], X_name: str,
                               y_name: str) -> None:
    """
    Plot the 3rd degree polynomial regression using matplot library
    :param coefficients: a list of coefficient for n-th degree X data, starting from degree of 0.
    :param X_data: up to n-th degrees of X data
    :param y_data: y values
    :param X_name: X feature name
    :param y_name: y result name
    :return: none
    """
    y_pred = []
    # set an X range for 3rd degree polynomial fit to predict y data
    X_range = np.linspace(0, 5, 1000)
    for X in X_range:
        y = coefficients[0] * X ** 0 + coefficients[1] * X ** 1 + coefficients[2] * X ** 2 + coefficients[3] * X ** 3
        y_pred.append(y)
    current_graph_title = y_name + " vs " + X_name
    # use tight layout and left adjustment to prevent y-axis label cutoff
    plt.tight_layout()
    plt.subplots_adjust(left=0.18)

    # plot each data point along with its best 3rd degree polynomial fit and save the figure
    plt.scatter(X_data, y_data, label=current_graph_title, color='b')
    min_y, max_y = min(y_data), max(y_data)
    plt.yticks(np.linspace(min_y, max_y, 10))
    min_X, max_X = min(X_data), max(X_data)
    plt.xticks(np.linspace(min_X, max_X, 5))
    plt.ylim(min_y, max_y)
    plt.xlim(min_X, max_X)
    third_degree_fit_label \
        = f"y={round(coefficients[3], 2)}x^3{round(coefficients[2], 2):+}x^2{round(coefficients[1], 2):+}x" \
          f"{round(coefficients[0], 2):+}"
    plt.plot(X_range, y_pred, label=third_degree_fit_label, color='r')
    plt.title(current_graph_title)
    plt.xlabel(X_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.savefig("3rd degree " + current_graph_title + ".png")
    plt.show()


def pca(data, normalize=True):
    """
    Computes the principal components of the given data
    :param data: the given data
    :param normalize: whether to whiten the data
    :return: the means, std, eigenvalues, eigenvectors, and the projected data as a tuple
    """
    # get the number of datapoint
    N = len(data)
    # assign the data to A as a numpy matrix
    A = np.matrix(data)[:, :-1]
    # assign to m the mean values of the columns of A
    m = np.mean(A, axis=0)
    print("mean is")
    print(m)
    # assign to D the difference A - m
    D = np.subtract(A, m)

    # if not normalized, this is the default value for non-whitened std
    std = np.matrix([1. for _ in range(A.shape[1])])
    # if normalize is true, compute the standard deviations of each column
    if normalize:
        std = np.std(D, axis=0)
    print("standard deviation is")
    print(std)
    # divide each column by its standard deviation vector
    D = np.divide(D, std)
    print("data after whitening:")
    print(D)
    # assign to U, S, V the result of running np.svd on D, with full_matrices=False
    U, S, VT = np.linalg.svd(D, full_matrices=False)
    print("U is")
    print(U)
    print("S is")
    print(S)
    # calculate explained_variance to understand how much of the variance can be explained by the particular
    # eigenvectors. Not required, just for learning purpose.
    explained_variance = S ** 2 / np.sum(S ** 2)
    print("explained variance is")
    print(explained_variance)
    print("eigenvector is")
    print(VT)

    # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
    # divided by the degrees of freedom (N - 1). The values are sorted
    eigenvalues = S ** 2 / (N - 1)
    print("eigenvalue is:")
    print(eigenvalues)

    # project the data onto the eigenvector. Treat V as a transformation matrix
    # and right-multiply it by D transpose. The eigenvectors of A are the rows of
    # V. The eigenvectors match the order of the eigenvalues.
    # use 0.95 as a cut-off value to determine how many dimensions to keep
    index = 0
    eigenvalues_sum = sum(eigenvalues)
    print(f"The eigenvalue sum is {eigenvalues_sum}")

    # use 0.95 cumulative eigenvalue sum as a cutoff to preserve the significant dimensions.
    eigenvalue_cutoff = eigenvalues_sum * 0.95
    cur_sum = 0
    for eigenvalue in eigenvalues:
        cur_sum += eigenvalue
        index += 1
        # if the current sum is greater or equal to the eigenvalue cumulative cutoff, terminate
        if cur_sum >= eigenvalue_cutoff:
            break
    print(f"The number of features to keep are: {index}")

    # calculate projected data
    projected_data = D @ VT.T[:, :index]
    print("projected data:")
    print(projected_data)

    # create a new data frame out of the projected data
    # return the means, standard deviations, eigenvalues, eigenvectors, and projected data
    return m, std, eigenvalues, VT, projected_data


if __name__ == "__main__":
    # get the file name and test percentage
    file_name = sys.argv[1]
    test_percent = int(sys.argv[2])

    # get the file prefix used to name training and test sets
    file_prefix = file_name.split('.')[0]

    # split the data file into training and test sets
    split_data(file_name, test_percent)

    # get the train and test data
    # the data is in the format of
    # [[x1,1, x2,1, x3,1, ..., y1],
    #  [x1,2, x2,2, x3,2, ..., y2]]
    # where each row contains a single value of each feature and its y value
    train_features, train_data = parse_data(file_prefix + " train set.csv")
    test_features, test_data = parse_data(file_prefix + " test set.csv")

    # turn the data into a numpy matrix and get its transpose so that each row contains all the value for a
    # single feature
    # the np_train_all_data is in the format of
    # [[x1,1, x1,2, x1,3, ..., x1,n],
    #  [x2,1, x2,2, x2,3, ..., x2,n],
    #  ...
    #  [y1, y2, y3, ........., yn]]
    np_train_data = np.array(train_data)
    np_train_all_data = np_train_data.T

    # plot all X features and y values using simple linear regression
    plot_simple_linear_regression(train_features, np_train_all_data)

    # get the X data matrix used to train multiple linear regression model, which does not have the y column
    np_train_multi_X_data = np_train_data[:, :-1]
    multiple_linear_regression(np_train_multi_X_data, np_train_all_data[len(np_train_all_data) - 1])

    num_of_col = len(np_train_all_data)

    # perform a 3rd degree polynomial regression on carat and price data
    train_carat_data = np_train_all_data[0]
    train_price_data = np_train_all_data[num_of_col - 1]
    linear_squared_cubic_X = compute_3rd_degree_X(train_carat_data)
    polynomial_model = polynomial_regression(linear_squared_cubic_X, train_price_data)
    plot_3rd_degree_polynomial(polynomial_model, train_carat_data, train_price_data, train_features[0],
                               train_features[num_of_col - 1])

    # test the PCA on the given test data
    file = open("pcatestdata.csv")
    file.readline()
    line = file.readline()
    data = []
    while line:
        each_line = []
        numbers = line.split(",")
        for number in numbers:
            each_line.append(float(number))
        data.append(each_line)
        line = file.readline()
    print("whitened data")
    pca(data)
    print("non-whitened data")
    pca(data, False)

    # get the projected data from the PCA on the training data and do multiple linear fit again using the projected data
    print("whitened diamond data")
    whitened_m, whitened_std, whitened_eigenvalues, whitened_eigenvectors, whitened_projected_data = pca(train_data)
    print("whitened projected data correlation")
    multiple_linear_regression(whitened_projected_data, np_train_all_data[len(np_train_all_data) - 1])

    # Extension Section
    print("extension section")

    np_test_data = np.array(test_data)
    np_test_multi_X_data = np_test_data[:, :-1]
    np_test_all_data = np_test_data.T
    test_price_data = np_test_all_data[num_of_col - 1]

    # Ridge regression
    print("using Ridge regression")
    ridge_reg = Ridge(alpha=0.1)
    ridge_reg.fit(np_train_multi_X_data, train_price_data)
    print(ridge_reg.coef_)
    print("using alpha=0.1 Ridge regression")
    print(f"training set: {ridge_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {ridge_reg.score(np_test_multi_X_data, test_price_data)}")
    ridge_reg = Ridge(alpha=1)
    ridge_reg.fit(np_train_multi_X_data, train_price_data)
    print(ridge_reg.coef_)
    print("using alpha=1 Ridge regression")
    print(f"training set: {ridge_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {ridge_reg.score(np_test_multi_X_data, test_price_data)}")
    ridge_reg = Ridge(alpha=10)
    ridge_reg.fit(np_train_multi_X_data, train_price_data)
    print(ridge_reg.coef_)
    print("using alpha=10 Ridge regression")
    print(f"training set: {ridge_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {ridge_reg.score(np_test_multi_X_data, test_price_data)}")
    ridge_reg = Ridge(alpha=100)
    ridge_reg.fit(np_train_multi_X_data, train_price_data)
    print(ridge_reg.coef_)
    print("using alpha=100 Ridge regression")
    print(f"training set: {ridge_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {ridge_reg.score(np_test_multi_X_data, test_price_data)}")

    # Lasso regression
    print("using Lasso regression")
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(np_train_multi_X_data, train_price_data)
    print(lasso_reg.coef_)
    print("using alpha=0.1 Lasso regression")
    print(f"training set: {lasso_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {lasso_reg.score(np_test_multi_X_data, test_price_data)}")
    lasso_reg = Lasso(alpha=1)
    lasso_reg.fit(np_train_multi_X_data, train_price_data)
    print(lasso_reg.coef_)
    print("using alpha=1 Lasso regression")
    print(f"training set: {lasso_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {lasso_reg.score(np_test_multi_X_data, test_price_data)}")
    lasso_reg = Lasso(alpha=10)
    lasso_reg.fit(np_train_multi_X_data, train_price_data)
    print(lasso_reg.coef_)
    print("using alpha=10 Lasso regression")
    print(f"training set: {lasso_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {lasso_reg.score(np_test_multi_X_data, test_price_data)}")
    lasso_reg = Lasso(alpha=100)
    lasso_reg.fit(np_train_multi_X_data, train_price_data)
    print(lasso_reg.coef_)
    print("using alpha=100 Lasso regression")
    print(f"training set: {lasso_reg.score(np_train_multi_X_data, train_price_data)}")
    print(f"test set: {lasso_reg.score(np_test_multi_X_data, test_price_data)}")
