# Numpy array 슬라이싱
# 다차원인 경우가 많기 때문에
# 각 차원별로 어떻게 슬라이싱할 것인지 명확히 해야함

import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

b = a[:2, 1:3]

print(b)

# Numpy 배열의 슬라이싱은 새로운 배열 생성 X
# 기존 배열에 대한 새로운 view 제공
# List 슬라이싱: 새로운 List 객체 생성
# 그러나 Numpy는 NOPE !!!

print(a[0, 1])
b[0, 0] = 77
print(a)
# 데이터는 똑같이.
# 무엇을 바꾸던 데이터는 바뀐다.

# 정수 인덱스와 슬라이스 인덱스 섞어서 함께 사용 가능

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)

row_r1 = a[1, :] # 1차원 배열
# 그냥 하나만 뽑아내면 rank 감소 (차원 감소)
row_r2 = a[1:2, :] # 2차원 배열, 행의 개수 하나
# 길이가 1인 것만 뽑아내면 차원 보존

print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print()
print(col_r2, col_r2.shape)