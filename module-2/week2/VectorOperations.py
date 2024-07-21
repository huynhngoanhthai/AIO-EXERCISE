import numpy as np

# a


def compute_vector_length(vector: np.array) -> float:

    len_of_vector = np.linalg.norm(vector)

    return len_of_vector


# b

def compute_dot_product(vector1: np.array, vector2: np.array) -> float:

    result = np.dot(vector1, vector2)

    return result


# c

def matrix_multi_vector(matrix: np.array, vector: np.array) -> np.array:

    result = np.dot(matrix, vector)

    return result


# d

def matrix_multi_matrix(matrix1: np.array, matrix2: np.array) -> np.array:

    result = np.dot(matrix1, matrix2)

    return result

# e


def inverse_matrix(matrix: np.array) -> np.array:
    result = np.linalg.inv(matrix)

    return result


# F
def compute_eigenvalues_eigenvectors(matrix: np.array):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    return eigenvalues, eigenvectors


# G
def compute_cosine(v1, v2):
    cos_sim = (compute_dot_product(v1, v2)) / \
        (compute_vector_length(v1)*compute_vector_length(v2))
    return cos_sim


x = np. array([1, 2, 3, 4])
y = np. array([1, 0, 3, 0])
result = compute_cosine(x, y)
print(round(result, 3))
