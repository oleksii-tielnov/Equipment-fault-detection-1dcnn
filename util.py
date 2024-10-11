import numpy as np
from numpy.linalg import norm
import json


def format_to_4_digits(num: int) -> str:
    return f"{num:04d}"


def get_vec(index) -> np.array:
    index = format_to_4_digits(index)

    with open(f".\\data\\T{index}.txt", 'r') as f:
        vec = [float(line.replace('\n', '')) for line in f.readlines()]
    
    return vec


def get_labels() -> list:
    with open(".\\data\\key.txt", 'r') as f:
        labels = [int(float(line.replace('\n', ''))) for line in f.readlines()]

    return labels


def get_etalons() -> tuple:
    with open('etalons.json', 'r') as file:
        data = json.load(file)

    return (data['e_0'], data['e_1'])


def get_zetas():
    with open("zetas_weights.json", 'r') as file:
        data = json.load(file)
    
    return [np.array(data['zeta_1']), np.array(data['zeta_2'])]


def update_zetas(zetas):
    zetas_dict = {"zeta_1": zetas[0], "zeta_2": zetas[1]}

    with open("zetas_weights.json", 'w') as file:
        json.dump(zetas_dict, file, indent=4)


def convolve(v: np.array , u: np.array) -> np.array:
    n, m = len(v), len(u)
    steps = n // m
    res = []
    
    i = 0
    for _ in range(steps):
        temp = sum([u[j-1] * v[i+j] for j in range(m)])
        res.append(temp)
        i += m

    return np.array(res)


def simlarity(u, v) -> float:
    return np.dot(u, v) / (norm(u) * norm(v))


def sigmoid(z):
    return 1 / (1 + np.exp(-z+4))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
