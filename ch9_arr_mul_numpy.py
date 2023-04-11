import numpy as np

# x(2) * w(2x3) = y(3)

x = np.array([1, 2])
w = np.array([[1, 3, 5], [2, 4, 6]])

print(w)
print(w.shape)
y = np.dot(x, w)
print(y)
