import numpy as np

from data_loader import load_data
from model import MLP
from util import get_label, get_vec, extract_features


def find_max(nums, length) -> list[int]:
    copy_nums = [num for num in nums]
    copy_nums.sort()
    copy_nums = copy_nums[-length:-1]

    max_i = 0
    max_indecies = []
    for i in range(len(copy_nums)):
        for j in range(len(nums)):
            if copy_nums[i] == nums[j]:
                max_i = j
                break

        max_indecies.append(max_i)

    return max_indecies


if __name__ == "__main__":
    net = MLP(mode="rand", loss="quadratic", sizes=[938, 20, 20, 10, 1])
    data = load_data(100)

    norms_w = []
    norms_b = []

    def loss(vec, label):
        return 0.5 * (net.forward(vec)[0] - label) ** 2
    
    losses = [loss(vec, label) for vec, label in data]
    min_indecies = find_max(losses, 10)

    for i in min_indecies:
        vec, label = extract_features(get_vec(i+1), step=100), get_label(i+1)
        nabla_w, nabla_b = net.backprop(vec, label)

        norms_w.append([
            np.linalg.norm(nw, 2) 
            for nw in nabla_w
            ])

        norms_b.append([
            np.linalg.norm(nb, 2) 
            for nb in nabla_b
            ])

    norms_w = np.concatenate(norms_w)
    norms_b = np.concatenate(norms_b)

    eta_w = np.mean(norms_w)
    eta_b = np.mean(norms_b)

    print(f"1/eta_w={1/eta_w}, 1/eta_b={1/eta_b}")
    print(f"eta_w={eta_w}, eta_b={eta_b}")
