import numpy as np

from util import get_vec, get_label, get_etalons, similarity


def load_data(amount: int) -> list:
    data = []
    e0, e1 = get_etalons()

    for _ in range(amount):
        i = np.random.randint(1, 1159)

        vec, label = get_vec(i), get_label(i)
        etalon = e0 if label == 0 else e1
        pair = [vec, etalon]

        data.append(pair)

    return data


if __name__ == "__main__":
    temp = load_data(10)
    e0, e1 = get_etalons()

    print(len(temp))

    print(temp[0][1])

    # print(similarity(temp[0][1], e0))
    # print(similarity(temp[0][1], e1))
