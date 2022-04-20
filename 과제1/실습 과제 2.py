# 실습 과제 2

f = open("cars.csv", 'r')
head = f.readline()
data = f.readlines()
f.close()

# 제조사가 Volkswagen인 가솔린 차들 중에서 형태가 Sedan인 자동차들 개수
count = 0
for i in range(len(data)):
    line = data[i].split(',')
    if 'Volkswagen' in line and 'Gas' in line and 'sedan' in line:
        count += 1
print("1번답: {} \n".format(count))

# 제조사가 BMW인 자동차들만 하나의 리스트로 저장한 후 제조년도에 대한 오름차순으로 정렬

def insert_sort(start, arr):
    x = arr[start]
    j = start - 1
    while j >= 0 and arr[j][-2] > x[-2]:
        arr[j + 1] = arr[j]
        j = j - 1
    arr[j + 1] = x
    if start == len(list) - 1:
        return
    insert_sort(start + 1, arr)
    return arr

list = []
for i in range(len(data)):
    line = data[i].split(',')
    if 'BMW' in line:
        list.append(line) # 뒤에 개행문자 없애는 법 모르겠음 

arr = insert_sort(0, list)

print("2번 답: \n")
for i in range(len(list)):
    print(arr[i])
