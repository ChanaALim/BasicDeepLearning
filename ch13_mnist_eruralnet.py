import os
import sys
# sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image
from ch7_sigmoid_ReLU import sigmoid
from ch11_softmax import softmax
import pickle


def get_data():
    # normalize=True : 0~255 범위 각 필셀의 값을 0.0~1.0 범위로 반환
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# sample_weight.pkl : 저장된 학습된 가중치 매개변수 (가중치, 편향 딕셔너리 변수로 저장)
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


# 각 레이블의 확률을 넘파이 배열로 반환
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    # train image range
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)    # 확률이 가장 높은 원소의 인덱스 반환
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy : " + str(float(accuracy_cnt) / len(x)))
    # print(len(x))

