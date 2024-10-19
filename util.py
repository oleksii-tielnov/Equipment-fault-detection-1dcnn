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


def get_label(index) -> int:
    with open(".\\data\\key.txt", 'r') as file:
        labels = [int(float(line.replace('\n', ''))) for line in file.readlines()]

    return int(labels[index - 1])


def values_init(filename: str="values.json") -> tuple:
    with open(filename, 'r') as file:
        data = json.load(file)

    weights = data['weights']
    biases = data['biases']
    
    w_arr = []
    b_arr = []

    for w, b in zip(weights, biases):
        w_mat = np.ndarray(shape=(len(w), len(w[0])), buffer=np.array(w))
        w_arr.append(w_mat)

        b_vec = np.ndarray(shape=(len(b), ), buffer=np.array(b))
        b_arr.append(b_vec)

    sizes = [w_arr[0].shape[1]]
    sizes = list(np.concatenate((sizes, [b.shape[0] for b in b_arr])))

    return (sizes, w_arr, b_arr)


def values_update(weights, biases, filename: str="values.json") -> None:
    w_list = []
    b_list = []

    num_layers = len(weights)

    for i in range(num_layers):
        matrix = []

        for j in range(weights[i].shape[0]):
            vec = list(weights[i][j])
            matrix.append(vec)

        w_list.append(matrix)

    for i in range(num_layers):
        vec = list(biases[i])
        b_list.append(vec)

    values_dict = {'weights': w_list, 'biases': b_list}

    with open(filename, 'w') as file:
        json.dump(values_dict, file, indent=4)


def similarity(u, v) -> float:
    return np.dot(u, v) / (norm(u) * norm(v))


def sigmoid(z) -> float:
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z) -> float:
    return sigmoid(z) * (1 - sigmoid(z))


def extract_features(vec, step: int) -> np.array:
    def max_pooling(u, s):
        return np.array([max(u[i:i+s]) for i in range(0, len(u), s)])

    vec = np.abs(vec)
    max_pooled = max_pooling(vec, step)
    return max_pooled - 3


if __name__ == "__main__":
    # w = [np.random.randn(10, 5) for _ in range(4)]
    # b = [np.random.randn(10, ) for _ in range(4)]

    # values_update(w, b)

    # w, b = values_init()
    s, w, b = values_init()

    print(s)
