
# 배열 연산

# 각 요소별로 동작

import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))
# 서로 대응되는 원소들끼리 연산

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))
# 그냥... 원소들끼리 곱하는 것

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))
# 배열 원소 전체를 루트로 씌워버릴 수도 있음

########## 행렬 곱 ##########

x = np.array([[1,2],
              [3,4]])
y = np.array([[5,6],
              [7,8]])

v = np.array([9,10])
w = np.array([11,12])

# shape이 맞아야 함
print(x.dot(y))
print(x.dot(v))

print(v.dot(w))
print(np.dot(v, w))

print(v @ w) # == v.dot와 같음

# Numpy는 sum, max, min, mean, std (표준편차)가 있음

x = np.array([[1,2],[3,4]])

print(np.sum(x))
print(np.sum(x, axis=0)) # 세로축
print(np.sum(x, axis=1)) # 가로축

print(np.max(x))
print(np.max(x, axis=1))
print(np.min(x))
print(np.min(x, axis=0))

print(np.mean(x)) # 평균
print(np.mean(x, axis=0))

print(np.std(x)) # 표준편차
print(np.std(x, axis=1))

# 전치행렬 -> transpose

print(x)
print("transpose\n", x.T)

v = np.array([[1,2,3]])
print(v )
print("transpose\n", v.T)

########## Broadcasting ############

# shape이 서로 다른 배열간의 연산을 하고 싶을 때는 ????

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(x + 2)
print(x * 2)

v = np.array([1, 0, 1])
y = x + v
print(y)
