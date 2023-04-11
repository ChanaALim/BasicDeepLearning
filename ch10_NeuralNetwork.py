import numpy as np

# 3층 신경망 구현 (입력층 2개, 은닉 1층 3개, 은닉 2층 2개, 출력층 2개)

# 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))     # exp(x) : exp(자연상수), x(지수)


# 분류 함수 (항등 함수)
def identiy_function(x):
    return x


# 가중치(W) 편향(b) 초기화
def init_network():
    # 딕셔너리 변수
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# 순전파(신호 순방향으로 이뤄짐)
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b3'], network['b3']

    a1 = np.dot(x, W1) + b1     # (1x2) * (2x3) + b(1x3) = (1x3)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2    # (1x3) * (3x2) + b(1x2) = (1x2)
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3    # (1x2) * (2x2) + b(1x2) = (1x2)
    y = identiy_function(a3)    # y = (1x2)

    return y


network = init_network()
x = np.array([1.5, 0.5])
y = forward(network, x)
print(y)
