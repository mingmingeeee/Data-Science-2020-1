# 네번째 container 타입: 튜플
# 소괄호 ()
# 리스트와 다른 점: 값 변경 불가능

d = {(x, x + 1): x for x in range(10)}
# key가 하나의 튜플임.
# list는 불가능하다.
t = (5, 6)
print(type(t))
print(d[t])
# tuple 5, 6을 키로 하는 원소
print(d[(1, 2)])
# tuple 1, 2를 키로 하는 원소