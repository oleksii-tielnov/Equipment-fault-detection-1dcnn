import numpy as np

from util import convolve, get_vec, similarity, get_etalons, get_zetas, sigmoid, sigmoid_prime, update_zetas, get_label
from data_loader import load_data


class OneDCNN:
    def __init__(self) -> None:
        self.zetas = get_zetas()

    def forward(self, a):
        for zeta in self.zetas:
            a = sigmoid(convolve(a, zeta)) 

        return a

    def SGD(self, amount, epochs, mini_batch_size, learning_rate):
        eta = learning_rate

        for _ in range(epochs):
            training_data = load_data(amount)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(amount)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

        update_zetas(self.zetas)

    def update_mini_batch(self, mini_batch, eta):
        h = len(self.zetas[0])
        g = len(self.zetas[1])

        nabla_zeta_1 = np.zeros(shape=(h, ))
        nabla_zeta_2 = np.zeros(shape=(g, ))

        for x, y in mini_batch:
            delta_nabla_zeta_1, delta_nabla_zeta_2 = self.backprop(x, y)

            nabla_zeta_1 += delta_nabla_zeta_1
            nabla_zeta_2 += delta_nabla_zeta_2
        
        self.zetas[0] -= (eta/ len(mini_batch)) * nabla_zeta_1
        self.zetas[1] -= (eta/ len(mini_batch)) * nabla_zeta_2

    def backprop(self, x, y):
        h = len(self.zetas[0])
        g = len(self.zetas[1])
        m = 75

        activation = x
        activations = [x]

        zs = []

        for zeta in self.zetas:
            z = convolve(activation, zeta)
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        def nabla_zeta_2():
            nabla_zeta_1 = []
            delta = (activations[2] - y)

            for n in range(g):
                wn = np.array([
                    sigmoid_prime(zs[1][k]) * activations[1][(k + 1)*g-n-1] 
                    for k in range(m)
                    ])
                nb_n = np.dot(delta, wn)

                nabla_zeta_1.append(nb_n)

            return np.array(nabla_zeta_1)
        
        def nabla_zeta_1():
            nabla_zeta_2 = []
            m = 93752 // h

            delta = []

            for k in range(m):
                r = k // g
                z_i = (g - 1 - k) % g
                
                delta_k = (activations[2][r] - y[r]) * sigmoid_prime(zs[1][r]) * self.zetas[1][z_i]
                delta.append(delta_k)

            delta = np.array(delta)

            for n in range(h):
                wn = np.array([
                    sigmoid_prime(zs[0][k]) * activations[0][(k + 1)*h-n-1] 
                    for k in range(m)
                    ])
                nb_n = np.dot(delta, wn)

                nabla_zeta_2.append(nb_n)

            return np.array(nabla_zeta_2)

        return nabla_zeta_1(), nabla_zeta_2()

    def evaluate(self, num) -> float:
        def covariance(x):
            return 1 if x >= 0.5 else 0

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

    # print(similarity(e1, net.forward(vec)))

    nabla_zeta_1, nabla_zeta_2 = net.backprop(vec, e0)

    net.update_mini_batch(load_data(10), 3)

    print(nabla_zeta_1)
    print(nabla_zeta_2)