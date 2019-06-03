import mnist_loader as loader
import network

train, validation, test = loader.load_data_wrapper()
nn = network.Network([784, 10])
nn.SGD(train, 30, 10, 3.0, test) # first attempt 
