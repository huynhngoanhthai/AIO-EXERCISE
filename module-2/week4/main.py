import numpy as np
import pandas as pd
import matplotlib . pyplot as plt
import seaborn as sns


def compute_mean(values: list) -> float:
    return sum(values) / len(values)


def compute_median(x):
    size = len(x)
    x = np.sort(x)
    print(x)
    if (size % 2 == 0):
        return (x[size // 2 - 1] + x[size // 2]) / 2
    else:
        return x[size // 2]


def compute_std(x):
    mean = compute_mean(x)
    variance = sum((value - mean)**2 for value in x) / len(x)
    return np.sqrt(variance)


def compute_correlation_coefficient(X, Y):
    N = len(X)
    sum_x = sum(X)
    sum_y = sum(Y)

    sum_xy = sum(x * y for x, y in zip(X, Y))
    sum_x2 = sum(x ** 2 for x in X)
    sum_y2 = sum(y ** 2 for y in Y)

    numerator = N * sum_xy - sum_x * sum_y
    denominator = np.sqrt((N * sum_x2 - sum_x ** 2)
                          * (N * sum_y2 - sum_y ** 2))

    if denominator == 0:
        return 0

    correlation_coefficient = numerator / denominator
    return correlation_coefficient


def get_data_csv():
    return pd.read_csv("advertising.csv")


def visualize(data_corr):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


if __name__ == "__main__":
    # for test case
    X = np . asarray([-2, -5, -11, 6, 4, 15, 9])
    Y = np . asarray([4, 25, 121, 36, 16, 225, 81])
    print(" Correlation : ", compute_correlation_coefficient(X, Y))
