from model import OneDCNN


net = OneDCNN()

net.SGD(amount=1000, epochs=2, mini_batch_size=10, learning_rate=0.5)
