from Main import train_gaussian_naive_bayes, prediction_iris, create_train_data_iris
import numpy as np


if __name__ == "__main__":
   # Example 1
    X = [6.3, 3.3, 6.0, 2.5]
    train_data = create_train_data_iris()
    y_unique = np.unique(train_data[:, 4])
    prior_probability, conditional_probability = train_gaussian_naive_bayes(
        train_data)
    pred = y_unique[prediction_iris(
        X, prior_probability, conditional_probability)]
    assert pred == "Iris-virginica"

    # Example 2
    X = [5.0, 2.0, 3.5, 1.0]
    train_data = create_train_data_iris()
    y_unique = np.unique(train_data[:, 4])
    prior_probability, conditional_probability = train_gaussian_naive_bayes(
        train_data)
    pred = y_unique[prediction_iris(
        X, prior_probability, conditional_probability)]
    assert pred == "Iris-versicolor"

    # Example 3
    X = [4.9, 3.1, 1.5, 0.1]
    train_data = create_train_data_iris()
    y_unique = np.unique(train_data[:, 4])
    prior_probability, conditional_probability = train_gaussian_naive_bayes(
        train_data)
    pred = y_unique[prediction_iris(
        X, prior_probability, conditional_probability)]
    assert pred == "Iris-setosa"
