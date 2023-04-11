import keyboard
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

print(os.getcwd())  #현재 작업 경로 출력
root_path = 'C:\\Users\\ami15\\Desktop\\annotation\\임찬아\\카메라'   #이미지 저장된 디렉터리 경로
img_path = os.listdir(root_path)    #해당 경로 내 파일 리스트로 저장

print(img_path)
print(type(img_path))


for i in img_path[0:]:
    print("i = " + i)
    img = imread(root_path + '\\' + i)

    plt.imshow(img)
    plt.show()
