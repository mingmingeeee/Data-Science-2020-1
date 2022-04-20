
from random import random

# x는 벡터값 x[0] = x x[1] = y
def f(x):
    return 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2

def df(x):
    dx = 4*x[0] + 2*x[1]
    dy = 2*x[0] + 2*x[1]
    return dx, dy
    # 두개의 값 한 번에 return 가능
    # (dx, dy) 투플로 리턴됨. ==> tuple == immutable list (값을 바꿀 수 없는 리스트

rho = 0.005
precision = 0.0000001
difference = 100

x = [random() for _ in range(2)]

# 를 풀어쓰면
# x = []
# for i in range(2):
#    tmp = random()
#    x.append(tmp)
# 이것이다.

while difference > precision:
    # dx, dy = df(x) -> 이렇게 받아와도 됨.
    # dr = [dx, dy] -> 이렇게 해도 됨.
    dr = df(x) # gradient 계산
    prev_x = x

    x = [x[i] - rho * dr[i] for i in range(2)]

    difference = (x[0]-prev_x[0])**2 + (x[1]-prev_x[1])**2
    # 길이의 차이 구하기
    print("x = {}, df = {}, f(x) = {:f}".format(x, dr, f(x)))