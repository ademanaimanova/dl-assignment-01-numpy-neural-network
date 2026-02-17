# C-Task
import numpy as np
X = np.array([[20,3,4],
              [15,5,3],
              [30,2,2],
              [25,4,1],
              [35,2,3]])
y = np.array([[18],[20],[22],[25],[30]])

# initializing weights
np.random.seed(42)
W1 = np.random.randn(3,3)   
b1 = np.zeros((1,3))
W2 = np.random.randn(3,1)   
b2 = np.zeros((1,1))

# activation functions
def relu(x): return np.maximum(0,x)
def relu_grad(x): return (x>0).astype(float)
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_grad(x): return sigmoid(x)*(1-sigmoid(x))

lr = 0.0001  
epochs = 10000
for epoch in range(epochs):
    # Forward
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = sigmoid(Z2)

    # Loss (MSE)
    loss = np.mean((y - A2)**2)

    # Backpropagation
    dA2 = 2*(A2 - y)
    dZ2 = dA2 * sigmoid_grad(Z2)
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # update weights
    W1 -= lr*dW1
    b1 -= lr*db1
    W2 -= lr*dW2
    b2 -= lr*db2

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, loss={loss:.4f}")

print("Final predictions:", A2)
print("Final W1:", W1)
print("Final W2:", W2)
print("Final b1:", b1)
print("Final b2:", b2)
