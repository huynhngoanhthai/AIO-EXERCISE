import numpy as np


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
