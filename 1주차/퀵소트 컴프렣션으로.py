
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

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,7,10,1,2,1]))