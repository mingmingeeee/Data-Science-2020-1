
# Numpy 배열
# 리스트와 유사하지만 처리속도 빠름
# -> 많은 데이터를 사용할 때 적합
# 다차원 배열 제공

import numpy as np

# 여러개의 데이터 저장
# 리스트를 이용해 Numpy 배열을 초기화 가능
# 서로 서로의 타입으로 변환 가능

a = np.array([1, 2, 3])
# 파이썬 리스트를 주고
# 이것을 초기값으로 하는 np.array
print(a)
# 출력된 것만 보면 리스트인지 np인지 구분이 안돼서 헷갈림
print(a[0], a[1], a[2])

print(type(a))
# np를 지정하는 방법
# 리스트를 이용해 지정할 수 있음
# 굳이 np 사용??? -> 처리속도 빠름

# np 속성

print(a.ndim) # number of dimension -> 배열의 차원, rank
# 1차원 배열이냐 2차원 배열이냐 ...

print(a.shape) # 행의 개수 * 열의 개수

print(a.dtype) # 리스트와 np의 차이
# 동일한 자료형의 데이터만 저장가능.
# 정수의 배열인지 실수의 배열인지 알 수 있음

a[0] = 5 # 배열 값 변경 가능
print(a)

b = np.array([[1,2,3],[4,5,6]]) # rank 2인 배열 생성
print(b)
print(b.ndim)
print(b.shape) # 소괄호로 묶여져 있으면 튜플
print(b[0, 0], b[0, 1], b[1, 0]) # 다차원 배열에서 각각의 값에 접근
print(b[(0, 0)], b[(0, 1)], b[(1, 0)]) # 다차원 배열에서 각각의 값에 접근
# 인덱스 값은 튜플로 주는게 사실은, 정석.
# 위의 방법도 ㄱㅊ

c = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(c.ndim)
print(c.shape)
print(c)
print(c[0, 1, 2], c[(0, 1, 2)])

################ 배열을 생성할 때 빈 값으로 만들고 나중에 값넣어줄 때
################ 특정한 값으로 초기화, 다양한 함수 제공

a = np.zeros((2, 2)) # 0으로 채워진 2*2 크기의 2차원 배열 생성
print(a)
print(a.dtype) # 지정하지 않으면 실수형으로 지정됨

b = np.ones((1, 2)) # 1로 채워진1*2 크기의 2차원 배열
print(b)

c = np.full((2, 2), 7) # 0도 아니고 1도 아닌 특정한 값으로 채우고 싶을 때
print(c)

############## Identity matrix
############## 사이즈는 d, 대각선만 1이고 나머지는 9
d = np.eye(3)
print(d)

e = np.random.random((2, 2)) # 랜덤 밸류로 채워짐
print(e)

###### Numpy 배열에는 동일한 타입의 값들이 저장됨
# 배열이 생성될 때 자료형을 스스로 추측
# 배열을 생성할 때 명시적으로 특정 자료형 지정할 수도 있음

x = np.array([1, 2]) # 추측
y = np.array([1.0, 2.0]) # 추측
z = np.array([1, 2], dtype=np.int32) # 지정
w = np.array([1, 2], dtype=np.float32) # 지정

print(x.dtype, y.dtype, z.dtype, w.dtype)