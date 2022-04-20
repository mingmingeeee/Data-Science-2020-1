# 실습 과제 1-2

def merge(left, right):
    i, j = 0, 0
    arr1 = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr1.append(left[i])
            i += 1
        else:
            arr1.append(right[j])
            j += 1
    while i < len(left):
        arr1.append(left[i])
        i += 1
    while j < len(right):
        arr1.append(right[j])
        j += 1
    return arr1

def merge_sort(arr):
    if len(arr) > 1:
        middle = len(arr)//2
        left = merge_sort(arr[:middle])
        right = merge_sort(arr[middle:])
        return merge(left, right)
    else:
        return arr


data = [ 3, 6, 8, 10, 1, 2, 1 ]
print(merge_sort(data))