from Main import create_train_data, compute_conditional_probability, compute_prior_probablity, get_index_from_value, prediction_play_tennis,  train_naive_bayes, np


def question_14() -> None:
    train_data = create_train_data()
    prior_probability = compute_prior_probablity(train_data)

    print("P(play tennis = No)", prior_probability[0])
    print("P(play tennis = Yes)", prior_probability[1])


def question_15() -> None:
    train_data = create_train_data()
    _, list_x_name = compute_conditional_probability(train_data)

    print("x1 =", list_x_name[0])
    print("x2 =", list_x_name[1])
    print("x3 =", list_x_name[2])
    print("x4 =", list_x_name[3])


def question_16() -> None:
    # Create train data and compute conditional probabilities
    train_data = create_train_data()
    _, list_x_name = compute_conditional_probability(train_data)

    # Get the unique values for the "Outlook" feature
    outlook = list_x_name[0]

    # Find indices for specific values in "Outlook"
    i1 = get_index_from_value("Overcast", outlook)
    i2 = get_index_from_value("Rain", outlook)
    i3 = get_index_from_value("Sunny", outlook)

    print("Index of 'Overcast' in outlook:", i1)
    print("Index of 'Rain' in outlook:", i2)
    print("Index of 'Sunny' in outlook:", i3)


def question_17() -> None:
    # Main code
    train_data = create_train_data()
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)

    # Compute P("Outlook"="Sunny"|Play Tennis="Yes")
    x1 = get_index_from_value("Sunny", list_x_name[0])
    print("P('Outlook'='Sunny'|Play Tennis='Yes') =",
          np.round(conditional_probability[0][x1, 1], 2))


def question_18() -> None:
    train_data = create_train_data()
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)

    x1 = get_index_from_value("Sunny", list_x_name[0])
    print("P('Outlook'='Sunny'|Play Tennis='No') =",
          np.round(conditional_probability[0][x1, 0], 2))

    train_data = create_train_data()
    prior_probability, conditional_probability, list_x_name = train_naive_bayes(
        train_data)
    test = ['Sunny', 'Cool', 'High', 'Strong']
    prediction = prediction_play_tennis(
        test, list_x_name, prior_probability, conditional_probability)
    print("Prediction for test {}: {}".format(test, prediction))


def question_19() -> None:
    X = ['Sunny', 'Cool', 'High', 'Strong']
    data = create_train_data()
    prior_probability, conditional_probability, list_x_name = train_naive_bayes(
        data)
    pred = prediction_play_tennis(
        X, list_x_name, prior_probability, conditional_probability)

    if pred == 1:
        print("Ad should go!")
    else:
        print("Ad should not go!")


if __name__ == "__main__":
    print("Log answer")

    question_14()
    question_15()
    question_16()
    question_17()
    question_18()
    question_19()
