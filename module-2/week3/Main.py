import numpy as np
import pandas as pd


# Create training data
def create_train_data():
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'no'],
        ['Sunny', 'Hot', 'High', 'Strong', 'no'],
        ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'no'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'yes']
    ]
    return np.array(data)


# Compute prior probabilities
def compute_prior_probablity(train_data):
    _, counts = np.unique(train_data[:, -1], return_counts=True)
    prior_probability = counts / train_data.shape[0]
    return prior_probability


# Compute conditional probabilities
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    for i in range(0, train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        x_conditional_probability = np.zeros((len(x_unique), len(y_unique)))

        for j, x_val in enumerate(x_unique):
            for k, y_val in enumerate(y_unique):
                count_x_and_y = np.sum(
                    (train_data[:, i] == x_val) & (train_data[:, -1] == y_val))
                count_y = np.sum(train_data[:, -1] == y_val)
                if count_y != 0:
                    x_conditional_probability[j, k] = count_x_and_y / count_y
                else:
                    x_conditional_probability[j, k] = 0

        conditional_probability.append(x_conditional_probability)

    return conditional_probability, list_x_name


# Get index of feature value
def get_index_from_value(feature_name, list_features):
    feature_name = feature_name.strip()
    return np.nonzero(list_features == feature_name)[0][0]


# Train Naive Bayes Model
def train_naive_bayes(train_data):
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probablity(train_data)

    # Step 2: Calculate Conditional Probability
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)

    return prior_probability, conditional_probability, list_x_name


# Prediction
def prediction_play_tennis(x, list_x_name, prior_probability, conditional_probability):
    x1 = get_index_from_value(x[0], list_x_name[0])
    x2 = get_index_from_value(x[1], list_x_name[1])
    x3 = get_index_from_value(x[2], list_x_name[2])
    x4 = get_index_from_value(x[3], list_x_name[3])

    p0 = prior_probability[0] * conditional_probability[0][x1, 0] * conditional_probability[1][x2,
                                                                                               0] * conditional_probability[2][x3, 0] * conditional_probability[3][x4, 0]
    p1 = prior_probability[1] * conditional_probability[0][x1, 1] * conditional_probability[1][x2,
                                                                                               1] * conditional_probability[2][x3, 1] * conditional_probability[3][x4, 1]

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


# Create training data for Iris dataset
def create_train_data_iris():
    data = pd.read_csv('data/iris.data.txt', header=None)
    data.columns = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width', 'class']
    return data.values


# Train Naive Bayes Model for Iris dataset
def train_gaussian_naive_bayes(train_data):
    prior_probability = compute_prior_probablity(train_data)

    classes = np.unique(train_data[:, -1])
    mean_var = {}
    for c in classes:
        data_c = train_data[train_data[:, -1] == c][:, :-1].astype(float)
        mean_var[c] = {
            'mean': np.mean(data_c, axis=0),
            'var': np.var(data_c, axis=0)
        }

    return prior_probability, mean_var


# Prediction for Iris dataset
def prediction_iris(x, prior_probability, mean_var):
    classes = list(mean_var.keys())
    posteriors = []

    for c in classes:
        mean = mean_var[c]['mean']
        var = mean_var[c]['var']
        prior = prior_probability[classes.index(c)]

        likelihood = np.prod(1 / np.sqrt(2 * np.pi * var)
                             * np.exp(- (x - mean) ** 2 / (2 * var)))
        posterior = prior * likelihood
        posteriors.append(posterior)

    return np.argmax(posteriors)
