import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) 
y = np.array(([92], [86], [89]), dtype=float) 

X=X/9
y=y/100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)

epoch, eta = 1000, 0.2  
input_neurons, hidden_neurons, output_neurons = 2, 3, 1  

np.random.seed(42)  
wh = np.random.uniform(size=(input_neurons, hidden_neurons))  
bh = np.random.uniform(size=(1, hidden_neurons)) 
wout = np.random.uniform(size=(hidden_neurons, output_neurons)) 
bout = np.random.uniform(size=(1, output_neurons))  

for _ in range(epoch):
    h_act = sigmoid(np.dot(X, wh) + bh)  
    output = sigmoid(np.dot(h_act, wout) + bout)  

    d_output = (y - output) * sigmoid_grad(output)
    d_hidden = d_output.dot(wout.T) * sigmoid_grad(h_act)

    wout += h_act.T.dot(d_output) * eta 
    wh += X.T.dot(d_hidden) * eta  
    bout += np.sum(d_output, axis=0, keepdims=True) * eta  
    bh += np.sum(d_hidden, axis=0, keepdims=True) * eta  

print("Normalized Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", output)
