import os
import sys
# sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image


def img_show(img):
    # fromarray() : 넘파이로 저장된 이미지 데이터 PIL용 데이터 객체로 변환
    pil_img = Image.fromarray(np.uint(img))
    pil_img.show()


# flatten=True : 1차원 넘파이 배열로 저장됨
# train_img, train_label, test_img, test_label
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print("train num : ", label)

print("numpy array image shape : ", img.shape)
# 넘파이 배열로 저장된 이미지를 28x28 크기로 변형
# reshape() : 넘파이 배열 형상 변환
img = img.reshape(28, 28)
print("image reshape : ", img.shape)

img_show(img)

