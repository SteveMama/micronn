import random

from calculate import Value


class Neuron:

    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):

        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        output = activation.tanh()
        return output


class Layer:

    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs



class MLPerceptron:

    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i] , sz[i+1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x