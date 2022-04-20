# 배열의 Reshape
# 원소가 유지되는 경우 자유롭게 다른 shape로 변경 가능함
# (3, 4) 크기의 2차원 배열은
# (2, 6) 크기의 2차원 배열이나
# 길이가 12인 1차원 배열
# 혹은 (2, 2, 3) 크기의 3차원 배열로도 변경할 수 있음

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(arr)

arr1 = np.reshape(arr, 12) # arr1 = arr.reshape(12)
# 원래 배열을 매개변수로 줘야함
print(arr1)

arr2 = np.reshape(arr,(2, 6))
print(arr2)

arr3 = np.reshape(arr, (2, 2, 3))
print(arr3)
# shape은 튜플로 표시...
# arr.reshape(2, 6)으로 할 수도 있음...

# reshape은 배열을 복사하여 새로운 배열을 생성하는 것이 아니라
# 존재하는 배열에 대한 새로운 view를 제공하는 것

arr3[0, 0, 0] = 99
print(arr)
print(arr3)

# 데이터는 보존되고 (똑같고), view만 달라짐

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
newarr = arr.reshape(2, 2, -1) # -1: 나는 모르겠으니 니가 알아서 지정해라
print(newarr)
# 길이는 똑같아야하기 때문에 저절로 3이 됨.
# 두개를 비워논다면? -> 에러. 어떻게 잡아야 하는지 모름

# 가장 자주 사용하는 reshape: 다차원 배열을 1차원 배열로 reshape

arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1) # 1차원은 어차피..정하지 않아도 ㄱㅊ
print(newarr)

# ravel 은 (arr), arr. 둘 다 가능
print(np.ravel(arr)) # 다차원 배열을 일차원 배열로 늘어뜨리는 함수
print(arr.ravel())

# print(np.flatten(arr)) 은 제공 X
print(arr.flatten()) # 또 똑같은 기능을 제공하는 함수