
def partition(arr):
    x = arr[len(arr)-1] # pivot 정해줌 -> 배열의 마지막 값
    i = -1
    for j in range(len(arr)-1):
        if arr[j] <= x:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[len(arr)-1] = arr[len(arr)-1], arr[i+1]
    return i+1

def quicksort(arr):
    if len(arr) <= 1:
        return arr

    q = partition(arr) # pivot 인덱스값 리턴
    left = arr[:q]
    right = arr[q+1:]

    return quicksort(left) + [arr[q]] + quicksort(right)

data = [ 3, 6, 8, 10, 1, 2, 1]
print(quicksort(data))