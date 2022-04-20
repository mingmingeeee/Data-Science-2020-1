# 실습 과제 1-1

def insert_sort(start, arr):
    x = arr[start]
    j = start - 1
    while j >= 0 and arr[j] > x:
      arr[j + 1] = arr[j]
      j = j - 1
    arr[j + 1] = x
    if start == len(data) - 1: # 끝까지 돌아서 검사 마치면 return
        return
    insert_sort(start + 1, arr)
    return arr # 최종적으로 arr 리턴

data = [ 3, 6, 8, 10, 1, 2, 1 ]
print(insert_sort(0, data))