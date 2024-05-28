import math
import random
class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda:None
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0* out.grad
            other.grad += 1.0* out.grad
        out._backward = _backward
        return out

    def __radd__(self,other):
        return self + other

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self,other):
        return self * other

    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,),'**')
        def _backward():
            self.grad += (other*(self.data**(other -1)))*out.grad
        out._backward  = _backward
        return out

    def __neg__(self):
        return self * -1
    
    def __sub__(self,other):
        return self + (-other)

    def __rsub__(self,other):
        return self - other

    def __truediv(self,other):
        out = Value(self.data * (other**-1),(self,other),'/')
        return out

    def log(self):
        out = Value(math.log(self.data),(self,other),'log')
        def _backward():
            self.grad += out.grad / (self.data)
        out._backward = _backward
        return out


    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
        out = Value(t,(self,),label="tanh")
        def _backward():
            self.grad += (1 - out.data**2)*out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        t = 1 / (1 + math.exp(-x))
        out = Value(x,(self,), label="sigmoid")
        def _backward():
            self.grad += (out.data *(1 - out.data)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0,(self,),label="relu")
        def _backward():
            self.grad += (1 if self.data > 0 else 0)*out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data),(self,),label="exp")
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out


    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set()
        def toposort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    toposort(child)
                topo.append(v)
        toposort(self)  
        for node in reversed(topo):
            node._backward()
        