
from random import random

def f(x):
    return x**4 - 12.0*x**2 - 12.0*x # ** 는 제곱

def df(x):
    return 4 * x**3 - 24 * x - 12

rho = 0.005 # rho 함수,.^^*
precision = 0.000000001
difference = 100 # while문 조건 때문에 그냥 아무거나 넣은 초기값
x = random()

while difference > precision:
    dr = df(x)
    prev_x = x
    x = x - rho * dr
    difference = abs(prev_x - x) # abs는 절대값
    # 차이가 precision부터 작아지면,
    # 차이가 너무 미미하다는 것을 나타냄.
    # 미분계수가 0에 가깝다는 뜻. -> 극소점에 거의 도달했다는 뜻
    print("x = {:f}, df = {:10.6f}, f(x) = {:f}".format(x, dr, f(x)))
