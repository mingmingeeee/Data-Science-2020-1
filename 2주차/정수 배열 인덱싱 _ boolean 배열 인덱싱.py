
# 다차원 배열에 대한 정수 배열 인덱싱
# rank개의 인덱스 배열로 제공되고 모든 인덱스 배열들이 동일한 shape 가짐
import numpy as np

y = np.arange(35).reshape(5, 7)
print(y)

print(y[np.array([0,2,4]), np.array([0,1,2])])
#                 세로축              가로축
#                [0,0] [2,1] [4,2]의 값을 뽑아 만들어짐
# 원본 배열의 rank개의 배열이 인덱스로 주어지고 동일한 shape인 경우
# 결과의 shape은 인덱스 shape과 동일함

print(y[np.array([[0,1],[2,3]]), np.array([[4,5], [2,3]])])



############# 인덱스 배열들이 서로 동일한 shape이 아닐 경우에는 그들을 동일한 shape으로 만들기 위해
############# broadcast를 시도
############# broadcast: 동일한 shape으로 만들기 위해 확장시키는 것.


# 원래 배열의 rank보다 적은 개수의 인덱스 배열 사용하여 인덱싱

# 배열 y는 2차원 배열. rank: 2
# 인덱스 - 1차원 배열
print(y[np.array([0,1,4])])
# 0행, 1행, 4행 만 뽑아 배열을 만듬

print(y[[0,1,4]]) # 로 해도 됨... ! ! !
# 내가 원하는 행들만 뽑고, 중복으로 뽑을 수도 있고, 행들의 순서도 뒤바꿀 수도 있다.

#################### 불리언 배열 인덱싱 (Boolean array indexing)

a = np.array([[1,2],
              [3,4],
              [5,6]])

bool_idx = (a > 2)
# == 각각의 원소들에 대해 '2보다 크다' 라는 조건을 검사하여
# 그 boolean 값들을 모아 배열로 만들어준다.
print(bool_idx)

print(a[bool_idx]) # boolean 배열에서 true로 지정된 원소들만 쭉쭉 뽑아 만들어줌
print(a[a > 2]) # 이렇게 해도 연산의 결과는 어차피 boolean 값이 되니까 똑같다.
