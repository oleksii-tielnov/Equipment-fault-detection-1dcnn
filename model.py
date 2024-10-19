import numpy as np

from util import get_vec, sigmoid, sigmoid_prime, extract_features, values_init, values_update


class MLP:
    def __init__(self, sizes: list[int], mode: str) -> None:
        match mode:
            case "rand":
                self.num_layers = len(sizes)
                self.sizes = sizes
                self.biases = [np.random.randn(y, ) for y in sizes[1:]]
                self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            case "init":
                self.sizes, self.weights, self.biases = values_init()
                self.num_layers = len(self.sizes)
            case _:
                raise ValueError("Incorrect mode")

    def forward(self, a) -> np.array:
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs: int, mini_batch_size: int, learning_rate: float) -> None:
        eta = learning_rate

        for _ in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

        # save weights and biases

    def update_mini_batch(self, mini_batch, eta) -> None:
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_weights, delta_nabla_biases = self.backprop(x, y)

            nabla_weights = [nw + dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
            nabla_biases = [nb + dnb for nb, dnb in zip(nabla_biases, delta_nabla_biases)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_weights)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_biases)]

    def backprop(self, x, y) -> tuple:
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        zs = []
        activation = x
        activations = [x]

        #region forward pass
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)
        #endregion

        #region backward pass
        delta = np.multiply((activations[-1] - y), sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2])

        for l in range(2, self.num_layers):
            w = self.weights[-l+1]

            z = zs[-l]
            spz = sigmoid_prime(z)

            delta = np.dot(w.T, delta) * spz
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1])
        #endregion

        return (nabla_w, nabla_b)
    
    def evaluate(self, num) -> float:
        pass


if __name__ == "__main__":
    vec = get_vec(1)
    vec = extract_features(vec, 100)

    net = MLP([938, 20, 20, 10, 1])

    net.backprop(vec, 0)

    # print(net.forward(vec))
