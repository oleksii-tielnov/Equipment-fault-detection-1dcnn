from model import MLP
from data_loader import load_data


net = MLP()

net.SGD(load_data(1150), epochs=10, mini_batch_size=10, learning_rate=1)
