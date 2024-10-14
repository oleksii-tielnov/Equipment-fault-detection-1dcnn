from model import OneDCNN


net = OneDCNN()

net.SGD(amount=100, epochs=5, mini_batch_size=5, learning_rate=0.5)
