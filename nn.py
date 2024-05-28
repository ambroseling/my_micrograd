from engine import Value
import random
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self,x):
        # out = Value(0.0)
        # for i in range(len(x)):
        #     out += self.w[i]*x[i] 
        # out += self.b
        # act = out.tanh
        #--- OR (much cleaner way) ---
        act = sum((wi * xi for wi,xi in zip(self.w,x)),self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self,in_neurons,out_neurons):
        self.neurons = [Neuron(in_neurons) for _ in range (out_neurons)]
    def __call__ (self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self,in_neurons,out_neurons):
        sz = [in_neurons] + out_neurons
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(out_neurons))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [neuron for layer in self.layers for neuron in layer.parameters()]
