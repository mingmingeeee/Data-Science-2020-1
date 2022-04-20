
# List Comprehension

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)

nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums] # nums의 모든 원소를 제곱한 리스트
print(squares)

nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0] # 짝수에만 제곱
print(even_squares)

words = ['hello', 'python', 'and', 'they', 'are', 'reality', 'fun']
selections = [str for str in words if 'e' in str]
print(selections)
longstrs = [str.capitalize() for str in words if len(str) > 4]
print(longstrs) # capitalize() :: 첫 글자만 대문자

