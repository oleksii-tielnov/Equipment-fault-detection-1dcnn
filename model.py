import numpy as np

from util import convolve, get_vec, similarity, get_etalons, get_zeta, sigmoid, sigmoid_prime, update_zeta, get_label, flip
from data_loader import load_data


class OneDCNN:
    def __init__(self) -> None:
        self.zeta = get_zeta()

    def forward(self, a) -> np.array:
        a = sigmoid(convolve(a, self.zeta)) 
        return a

    def SGD(self, amount: int, epochs: int, mini_batch_size: int, learning_rate: float) -> None:
        eta = learning_rate

        for _ in range(epochs):
            training_data = load_data(amount)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, amount, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

        update_zeta(list(self.zeta))

    def update_mini_batch(self, mini_batch, eta) -> None:
        h = len(self.zeta)

        nabla_zeta_1 = np.zeros(shape=(h, ))

        for x, y in mini_batch:
            delta_nabla_zeta_1 = self.backprop(x, y)

            nabla_zeta_1 += delta_nabla_zeta_1
        
        self.zeta -= (eta / len(mini_batch)) * nabla_zeta_1

    def backprop(self, x, y) -> tuple[np.array]:
        h = len(self.zeta)
        m = len(x) // h

        a = x
        z = convolve(a, self.zeta)
        new_a = sigmoid(z)

        delta = np.multiply((new_a - y), sigmoid_prime(z))

        def nabla_zeta_1() -> np.array:
            A = []

            for j in range(h):
                activation = np.array([a[k*h+j] for k in range(m)])    
                A.append(activation)

            return np.array(flip(np.dot(A, delta)))

        return nabla_zeta_1()

    def evaluate(self, num) -> float:
        def covariance(x):
            return 0 if x >= 0.5 else 1

        e0, e1 = get_etalons()
        detections = 0

        for i in range(1, num):
            vec, label = get_vec(i), get_label(i)
            prediction = covariance( similarity(e0, self.forward(vec)) )
            detections += 1 if prediction == label else 0

        accuracy = detections / num
        return accuracy


if __name__ == "__main__":
    net = OneDCNN()

    vec = get_vec(3)
    e0, e1 = get_etalons()

    # nabla_zeta_1 = net.backprop(vec, e0)

    # print(nabla_zeta_1, len(nabla_zeta_1))

    # net.update_mini_batch(load_data(10), 3)

    # print(nabla_zeta_1)
    # print(nabla_zeta_2)

    print(net.evaluate(1159))

    # print(net.forward(vec))
    # print(similarity(net.forward(vec), e0))
