from model import OneDCNN


net = OneDCNN()

net.SGD(amount=1150, epochs=10, mini_batch_size=10, learning_rate=0.25)
