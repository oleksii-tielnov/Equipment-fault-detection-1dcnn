import numpy as np

from util import get_vec, get_label, similarity, extract_features


def load_data(amount: int) -> list:
    data = []

    for i in range(1, amount):
        step = 100
        vec, label = extract_features(get_vec(i), step), get_label(i)
        pair = [vec, label]

        data.append(pair)

    return data


if __name__ == "__main__":
    n = 20
    step = 100
    vecs = load_data(n)

    # ef_vecs = [extract_features(vecs[i][0], step) for i in range(len(vecs))]

    print(vecs[0][0])
    