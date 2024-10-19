import numpy as np
from numpy.linalg import norm
import json


def format_to_4_digits(num: int) -> str:
    return f"{num:04d}"


def get_vec(index) -> np.array:
    index = format_to_4_digits(index)

    with open(f".\\data\\T{index}.txt", 'r') as file:
        vec = [float(line.replace('\n', '')) for line in file.readlines()]
    
    return vec


def get_label(index) -> list[int]:
    with open(".\\data\\key.txt", 'r') as file:
        labels = [int(float(line.replace('\n', ''))) for line in file.readlines()]

    return int(labels[index - 1])


def values_init() -> list:
    pass


def values_update(weights, biases):
    w_arr = []
    b_arr = []

    num_layers = len(weights)

    for i in range(num_layers):
        matrix = []

        for j in range(weights[i].shape[0]):
            vec = list(weights[i][j])
            matrix.append(vec)

        w_arr.append(matrix)

    for i in range(num_layers):
        vec = list(biases[i])
        b_arr.append(vec)

    values_dict = {'weights': w_arr, 'biases': b_arr}

    with open("values.json", 'w') as file:
        json.dump(values_dict, file, indent=4)


def similarity(u, v) -> float:
    return np.dot(u, v) / (norm(u) * norm(v))


def sigmoid(z) -> float:
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z) -> float:
    return sigmoid(z) * (1 - sigmoid(z))


def extract_features(vec, step: int) -> np.array:
    def max_pool(u, s):
        return np.array([max(u[i:i+s]) for i in range(0, len(u), s)])

    vec = np.abs(vec)
    max_pooled = max_pool(vec, step)
    return max_pooled - 3


if __name__ == "__main__":
    w = [np.random.randn(10, 5) for _ in range(4)]
    b = [np.random.randn(10, ) for _ in range(4)]

    values_update(w, b)
