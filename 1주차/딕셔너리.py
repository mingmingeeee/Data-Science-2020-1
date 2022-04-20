
# 딕셔너리 :: (key, value) 쌍 저장 {}
# key, value 쌍이 하나의 집함이 되는 것
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat']) # cat 이라는 key에 해당하는 value(값)이 'cute'이다.
print('cat' in d)

d['fish'] = 'wet'
print(d['fish']) # 딕셔너리 추가하는 법
print(d)

# print(d['monkey']) 존재하지 않는 키로 하면 에러
print(d.get('monkey')) # 없으면 None 반환


del d['fish'] # delete
print(d.get('fish', 'N/A'))

d = {'Person':2, 'cat':4, 'spider':8}
for animal, legs in d.items(): # key, value
    print('A {} has {} legs'.format(animal, legs))

nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0} 
print(even_num_to_square)