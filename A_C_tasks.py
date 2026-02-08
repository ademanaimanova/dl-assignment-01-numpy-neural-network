import numpy as np

# A.1 Matrix Multiplication (Resource Allocation)
Resources_Matrix = np.array([[10, 20],
                             [30, 40],
                             [50, 60]])
Allocation_Factors = np.array([[1, 2],
                               [3, 4]])

Result = Resources_Matrix.dot(Allocation_Factors)
print("A.1 Result:\n", Result)

# A.2 Element-wise Operations (Production Tracking)
Shift_A = np.array([10, 12, 11, 13, 14, 15, 16])
Shift_B = np.array([9, 11, 10, 12, 13, 14, 15])
Total_Production = Shift_A + Shift_B
print("A.2 Total Production:", Total_Production)

# A.3 Activation Function (Sigmoid in Sales Forecasting)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

test_values = np.array([-1, 0, 1, 2])
print("A.3 Sigmoid:", sigmoid(test_values))

# A.4 Gradient Calculation (Learning Adjustment)
def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

print("A.4 Sigmoid Gradient:", sigmoid_grad(test_values))
