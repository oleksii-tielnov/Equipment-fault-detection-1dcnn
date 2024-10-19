from model import MLP


net = MLP()

net.SGD(amount=1150, epochs=10, mini_batch_size=10, learning_rate=1)
