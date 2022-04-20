
import numpy as np
import os
import cv2

# 파이썬에는 배열이 없음.
# numpy에서 배열 제공해줌. 그러므로 ndarray

# opencv를 이용하여 이미지를 프로그램 내로 load하고 간단한 조작하는 것을 연습
# 프로그램 내에서

path = 'c:\\Users\\KANGMINJEONG\\Desktop\\animal_images\\cat\\images-2.jpeg'
# img RGB 값. 픽셀의 값을 RGB라고 함

img = cv2.imread(path)
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# GRAYSCALE로... 즉 흑백 이미지로 읽은 것임 <- 2차원 배열.
# 컬러는 3차원 배열

cv2.imshow('Test Image', img)
cv2.waitKey(0) # 키보드가 눌리길 기다림
cv2.destroyAllWindows() # 눌리면 닫힘

cv2.imwrite('original.jpg', img) # img 저장

img = cv2.resize(img, (200, 100), interpolation=cv2.INTER_CUBIC) # resize
# opencv 에서는 (넓이, 높이)로 img를 읽어오지만
# python에서는 (높이, 넓이)로 표현됨
cv2.imwrite('original.jpg', img) # img 저장
"""
print(img.ndim)
print(img.shape)
print(img.dtype)
img_list = img.tolist
print(img.tolist()) # tolist:: python list로 변환하는
img_array = np.array(img_list) # nd.array:: array로 돌아옴
# list와 array는 서로 서로 변환 가능
print(img)
"""
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 순서를 RGB 순서로 바꾸는 것
# python에서는 다르게 읽어와서,,, COLOR_BGR2RGB로 순서 바꿔줌,,,
cv2.imwrite('converted.jpeg', img)

# 차트 그리기 위한 것
from matplotlib import pyplot as plt
# lib 설치 안했으면... alt+Enter 해서 하면 됨!!!
# pyplot 이라는 이름은 기니까 plt로 사용하겠다.. 이런 뜻임!!

plt.imshow(img)
plt.show()

# numpy에서 가능한 SLICING, RESHAPE #

########## SLICING ##########
"""
배열을 잘라내는 것것
"""

########## RESHAPE ##########
"""
배열 모양을 reshape 하는 것
예::))
3x4 배열 -> 4x3 배열
1 2 3       1 2 3 4
4 5 6  ->   4 5 6 null
reshape을 하고 나서도 순서는 유지됨
그 말은 곧,,,,
총 SIZE는 유지되지만
행과 열은 바뀜 !!!
"""
# reshape이 중요함...
# 왜냐...??? 신경망 만들려고...!!!
img_reshaped = img.reshape(200, 100, 3)
plt.imshow(img_reshaped)
plt.show()

### 1차원 레벨로 만들어주는 함수 :: falttened ###
flattened_imgae = img.ravel()

### 신경망안에서 이루어지는 연산들은 실수 연산들임 ###
### 그러므로 신경망에 입력하기 직전에 img를 실수로 변환해주어야함 ###
### 그것이 아래 함수임!!!!!!!! ###
img_f32 = np.float32(img)
# == img_f32 = flattened_image.astype(np.float32)
# 배열이름.astype(np.float32)

# 스칼라 값으로 나누어주면 배열안의 모든 값이 이 값으로 나눠짐
normalized_imge32 = img_f32/255
# 배열 각각의 원소에게 평균값을 빼주고
# std: 표준편차...
# 나누어주면 배열의 각각 원소에게 빼줌
# 평균이 0이 되고 음수 양수 분포 ... 평균 0, 표준편차 1
zero_centered_img = (img_f32 - np.mean(img_f32)/np.std(img_f32))
# 이미지의 명도나 contrast 는 값은 값을 가지게 함.
# 편차를 없애기 위함임...@@@

print(zero_centered_img)