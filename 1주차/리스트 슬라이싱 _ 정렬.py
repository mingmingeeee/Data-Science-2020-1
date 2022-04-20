# 리스트 슬라이싱

name = list(range(5))

name [2:4] = [8, 9]
print(name)

print(name[::2]) # strat : end : step -> step은 몇개씩 건너뛸지를 지정함
print(name[-1::-1])

a = [1, 2, 3, 4]
b = a # 문자열과 같이 같은 것을 동시에 참조함
c = a[:] # 슬라이싱하면 별개의 객체가 생성됨
b[0] = 0
c[1] = 0
print(a)
print(b)
print(c)

print('\n')

# 인덱스가 필요할 때

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print("%{} = {}".format(idx + 1, animal))

# 리스트의 정렬

print('\n')
cars = ['Ford', 'BMW', 'Volvo']
print(cars)
cars.sort() # 알파벳순으로
print(cars)
cars.sort(reverse=True) # 역순으로
print(cars)

# 길이 순서대로
def myFunc(n):
    return len(n)

cars = ['Ford', 'Mitss', 'BMW', 'VW']

cars.sort(key=myFunc)
print(cars)

# 원본 데이터 보존
# 원본 데이터를 sorting한 새로운 리스트 만들어야 할 때
result = sorted(cars)