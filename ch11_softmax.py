import numpy as np


def softmax_overflow(x):
    x_exp = np.exp(x)
    x_sum_exp = np.sum(x_exp)
    y = x_exp / x_sum_exp

    return y


def softmax(x):
    # 오버플로 방지
    # 입력 신호 중 최댓값 이용
    c = np.max(x)
    x_exp = np.exp(x - c)
    x_sum_exp = np.sum(x_exp)
    y = x_exp / x_sum_exp

    return y


if __name__ == "__main__":
    overflow = np.array([1010, 1000, 900])
    a = np.array([0.3, 2.9, 4.0])
    print("------over array output------")
    print(softmax_overflow(overflow))
    print(softmax(overflow))
    print(np.sum(softmax(overflow)))
    print("------normal array output------")
    print(softmax_overflow(a))
    print(softmax(a))
    print(np.sum(softmax(a)))

