import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)     # 두 입력 중 큰 값 반환


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = ReLU(x)
    plt.plot(x, y1, label="sigmoid")
    plt.plot(x, y2, linestyle="-.", label="ReLU")
    plt.ylim(-0.1, 1.1)
    plt.title('sigmoid & ReLU')
    plt.legend()
    plt.show()
