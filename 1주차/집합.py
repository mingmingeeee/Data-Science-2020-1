# 집합 -> 순서가 없는 서로 다른 원소들의 모임 -> {}
# 리스트 -> 순서가 있음

animals = {'cat', 'dog'}
print('cat' in animals)
print('fish' in animals)

animals.add('fish') # 집합에 새로운 원소 추가
print(len(animals)) # cat이 있는데 또 추가하면 아무일도 일어나지 않는다.
# 집합에는 중복된 원소가 없어야 하기 때문이다.
animals.remove('cat') # 집합에서 'cat' 없애기
print(len(animals))

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal)) # 순서가 없기 때문에 매번 순서가 다름

from math import sqrt
print({int(sqrt(x)) for x in range(30)})