import numpy as np

a = np.array([[1, 2, 3], [3, 4, 5]])
print(a.shape, np.ndim(a))  # np.ndim() : 배열의 차원의 수 반환
b = np.array([[5, 6], [7, 8], [9, 1]])
print(b.shape, np.ndim(b), "\n")

# 행렬 곱 연산
print("a x b =\n", np.dot(a, b))
print("b x a =\n", np.dot(b, a))
