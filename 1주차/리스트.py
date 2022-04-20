# List :: 파이썬에는 배열같은 존재
# 배열과 달리 크기변경이 가능하다. -> C에서는 크기가 정해져있었다.
# 서로 다른 자료형을 저장할 수도 있다.
# 배열은... 같은 자료형만 저장할 수 있었다.

empty = [] # List 정의
xs = [3, 1, 2]
print(len(xs))

# 뒤에서부터 거꾸로 indexing 가능
print(xs[-1]) # 가장 마지막 원소
print(xs[-2]) # 마지막에서 두번째
print(xs[-3]) # 마지막에서 세번째

# 리스트에 저장된 값 변경 가능

xs[2] = 'foo'
print(xs)

# 리스트에 새로운 원소 추가

xs.append('bar') # 맨 끝에
print(xs)

xs.insert(1, 'orange') # 원하는 인덱스에 추가
print(xs)

x = xs.pop() # 맨 끝이 떨어져나감
print(xs)
xs.remove(1) # '1'이라는 인덱스가 아닌 값을 찾아 삭제하는 것.
print(xs)
xs.remove(xs[1]) # 이렇게 하면 1에 있는 것이 삭제될 것
print(xs)
del xs[0] # 위치로 지우기
print(xs)

print('\n')

thislist = ["apple", "banana", "cherry"]
newlist = thislist
newlist[0] = 'orange'
# list 자체는 하나만 있는데
# 그 것을 thislist와 newlist가 동시에 참조할 수 있는걸 알 수 있음
print(thislist)

print('\n')

# 동시에 참조하지 않고 진짜 카피만 하는 방법1
mylist = thislist.copy()
mylist[0] = "melon"
print(thislist)
print(mylist)

print('\n')

# 방법2
thislist = ["apple", "banana", "cherry"]
mylist = list(thislist) # list() :: list 생성자
print(mylist)

print('\n')

list1 = ["a", "b", "c"]
list2 = [1, 2, 3]
list1 += list2 # list 두개 합치는 법
print(list1)

# == list.extend(list)

print('\n')

#리스트의 리스트는 2차원 배열과도 같은 역할을 함,,,

ys = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
print(ys)
print(ys[1][2])

for r in ys:
    print(r)