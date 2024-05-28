import os
from nn import Neuron,MLP
from engine import Value
from utils import draw_dot

# x1 = Value(1.0,label="x1")
# x2 = Value(2.0,label="x2")
# w1 = Value(-4.0,label="w1")
# w2= Value(5.0,label="w2")
# w3 = Value(3.0,label="w3")
# w4= Value(2.5,label="w4")
# w5= Value(4.1,label="w5")
# w6= Value(5.8,label="w6")
# w7= Value(8.0,label="w7")
# w8= Value(20.0,label="w8")
# w9= Value(11.0,label="w9")
# w10= Value(10.0,label="w10")

# o1 = x1*w1
# o1.label = "o1"
# o2 = x2 * w2
# o2.label = "o2"
# o2_r = o2.relu()
# o2_r.label = "o2_r"
# o1_r = o1.relu()
# o1_r.label = "o1_r"
# o3 = o1_r * w3 + o2 *w5
# o3.label = "o3"
# o3.label = "o3"
# o3_r = o3.relu()
# o3_r.label = "o3_r"
# o4 = o1_r *w4 + w6 * o2_r
# o4.label = "o4"
# o4_r = o4.relu()
# o4_r.label = "o4_r"
# o5 = o3_r*w7 + o4_r*w8
# o5.label = "o5"
# o5 = o5 + o1_r
# o6 = o2_r * w9 + w10*w1
# o6.label = "o6"
# y1 = o5.sigmoid()
# y1.label  = "y1"
# y2 = o6.sigmoid()
# y2.label  = "y2"


# print(y1)
# # print(y2)
# dot = draw_dot(y1)
# dot.render("sample")
# os.system("dot -Tpng sample -o sample.png")

#Running a neuron
# x = [Value(2.0),Value(3.0),Value(-4.0)]
# n = Neuron(3)
# o = n(x)
# print(o)


#Running MLP
# x = [2.0,3.0,-1.0]
# n = MLP(3,[4,4,1])
# out = n(x)
# print(out)

#Training MLP
x = [
[2.0,3.0,-1.0],
[3.0,-1.0,0.5],
[0.5,1.0,1.0],
[1.0,1.0,-1.0]
]
y = [1.0,-1.0,-1.0,1.0]
n = MLP(3,[4,4,1])
y_pred = [n(x[i]) for i in range(len(x))]
learning_rate = 0.0001
training_epochs = 100
#Loss calculation
for epoch in range(training_epochs):
    y_pred = [n(x[i]) for i in range(len(x))]
    loss =sum([(ygt - yout)**2 for ygt,yout in zip(y,y_pred)])
    print(f"Epoch {epoch}: loss = {loss.data}")
    n.zero_grad()
    loss.backward()
    for parameters in n.parameters():
        parameters.data += -parameters.grad* learning_rate