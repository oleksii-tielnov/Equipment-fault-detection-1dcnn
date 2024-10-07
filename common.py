def format_to_4_digits(num: int) -> str:
    return f"{num:04d}"


def get_vec(index):
    with open(f".\\zeroShot_vkiit\\T{index}.txt", 'r') as f:
        vec = [float(line.replace('\n', '')) for line in f.readlines()]
    
    return vec


def get_labels():
    with open(".\\zeroShot_vkiit\\key.txt", 'r') as f:
        labels = [float(line.replace('\n', '')) for line in f.readlines()]

    return labels


def convolve(v, u):
    n, m = len(v), len(u)
    steps = n // m
    res = []
    
    i = 0
    for _ in range(steps):
        temp = sum([u[j-1] * v[i+j] for j in range(m)])
        res.append(temp)
        i += m

    return res
