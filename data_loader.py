from common import get_vec, get_labels, format_to_4_digits


def load_data(amount: int):
    
    data = []
    labels = get_labels()
    for i in range(1, amount):
        new_i = format_to_4_digits(i)
        
        vec = get_vec(new_i)
        label = labels[i - 1]
    
        entity = [vec, label]
        data.append(entity)

    return data


if __name__ == "__main__":
    temp = load_data()

    print(type(temp[0]), len(temp[0]), len(temp))
