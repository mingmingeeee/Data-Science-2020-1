# 실습 과제 3

# 헤드 정보를 이용하여 데이터를 딕셔너리의 리스트로 변환하여 저장 ㅇ
# 각각의 딕셔너리는 하나의 자동차 표현
# Brand, Price, Body, Mileage, EngineV, Engine Type, Registration, Year, Model의 9개의 키

# Price > 20000 and Price < 50000
# Year > 2000
# Body sedan
# Engine Type Gas
# 공통적으로 가져야함
# 가격이 낮은 것부터 높은 순으로 정렬하여 출력

# 가격이 $20,000~$50,000 범위
# 제조 년도가 2000년 이후
# 형태가 sedan
# Gas 사용
# 가격이 낮은 것부터 높은 순으로 정렬하여 출력

# 모든 차량 제조사의 이름을 저장하는 집합(set) 생성후 출력 ㅇ

f = open('cars.csv', 'r')   # 파일을 open한다.
head = f.readline().split(',')         # 파일의 첫 라인을 읽는다. 첫 라인은 테이블의 head이다.
Key = head                   # Brand,Price,Body,Mileage,EngineV,Engine Type,Registration,Year,Model
data = f.readlines()          # 파일의 나머지 모든 라인을 읽어온다. 라인들의 리스트로 저장된다.
f.close()

tmp = data[0].split(',')
dict_cars = {Key[i]: [tmp[i]] for i in range(len(tmp))}

for line in range(len(data)):
    tmp = data[line].split(',')
    j = 0
    for i in Key:
        dict_cars[i].append(tmp[j])
        j += 1


# Price > 20000 and Price < 50000
# Year > 2000
# Body sedan
# Engine Type Gas
# 공통적으로 가져야함
# 가격이 낮은 것부터 높은 순으로 정렬하여 출력
# index 값 구해야하나...?

def insert_sort(start, arr1, arr2):
    x, y = arr1[start], arr2[start]
    j = start - 1
    while j >= 0 and arr1[j] > x:
        arr1[j + 1] = arr1[j]
        arr2[j + 1] = arr2[j]
        j = j - 1
    arr1[j + 1] = x
    arr2[j + 1] = y
    if start == len(arr) - 1:
        return
    insert_sort(start + 1, arr1, arr2)
    return arr

arr = []
index = []
k = 0
for i in range(len(dict_cars['Body'])):
    if dict_cars['Body'][i] == "sedan" and dict_cars['Engine Type'][i] == "Gas":
        if dict_cars['Price'][i] == 'NA':
            dict_cars['Price'][i] = 10000000
        a = float(dict_cars['Price'][i])
        if int(dict_cars['Year'][i]) >= 2000 and (a > 20000 and a < 50000):
            arr.append(a)
            index.append(i)

insert_sort(0, arr, index)

for i in index:
    print(data[i - 1])

# 제조사 set
Car_Brands = set(dict_cars['Brand'])
print("제조사: {}".format(Car_Brands))

