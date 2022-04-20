import numpy as np

# 배열 합치기

# 1. Concatenate 함수
# 특정 축을 따라서 합침

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5])

arr = np.concatenate((arr1, arr2))

print(arr)

arr1 = np.array([[1, 2],
                [3, 4]])
arr2 = np.array([[5, 6],
                 [7, 8]])

# 합쳐질 두 배열은 동일한 rank를 가져야 함
# 다차원이라면 동일한 길이도 가져야 함

# 다차원이 된다면 어떤 축으로 합칠 건지 정해줘야함
arr3 = np.concatenate((arr1, arr2), axis = 0)
print(arr3)
# axis = 0 : 새로축

arr4 = np.concatenate((arr1, arr2), axis = 1)
print(arr4)
# axis = 1 : 가로축

# 2. Stack 함수
# 쌓는 것
# Concatenate: 연결
# 1차원 배열을 쌓아서 2차원으로 만드는...
# 2차원 배열을 쌓아서 3차원으로 만드는...
# 합쳐진 배열들은 모두 동일한 shape이어야 함

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.stack((arr1, arr2), axis = 0)
print(arr3)
# 세로로

arr4 = np.stack((arr1, arr2), axis = 1)
print(arr4)
# 가로로