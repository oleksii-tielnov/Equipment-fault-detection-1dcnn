from matplotlib import pyplot as plt
import numpy as np


def rerange_num(num: int) -> str:
    digits = len(str(num))
    reranged = None
    
    match digits:
        case 1:
            reranged = f"000{num}"
        case 2:
            reranged = f"00{num}"
        case 3:
            reranged = f"0{num}"
        case 4:
            reranged = f"{num}"

    return reranged


def get_vec(index):
    with open(f".\\zeroShot_vkiit\\T{index}.txt", 'r') as f:
        vec = [float(line.replace('\n', '')) for line in f.readlines()]
    
    return vec


def get_labels():
    with open(".\\zeroShot_vkiit\\key.txt", 'r') as f:
        labels = [float(line.replace('\n', '')) for line in f.readlines()]

    return labels


def angle(vec_1, vec_2):
    return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))



l = get_labels()
for i in range(10, 30):
    new_index = rerange_num(i)
    
    vec = get_vec(new_index)
    t = np.linspace(0, 1, len(vec))

    plt.plot(t, vec)
    plt.xlabel("samples")
    plt.ylabel("amplitude")
    plt.title(f"T{new_index}, class={l[i]}")

    plt.savefig(f".\\plots\\T{new_index}_plot.png")

    plt.clf()
