str = '1, 2, 3, 4'

print(str.split(','))
print(str.split(',', maxsplit = 2)) # maxsplit: 2개만 토크나이징해라
# 만약 1, 2, 3,,, 4, 라면 3의 뒤에 길이가 0인 str, 4의 뒤에 길이가 0인 str이 저장된다.
# C의 strtok 처럼 반복되는 것을 무시하지 않는다.

A = "my name is blue, thanks"
print(A.replace("blue", "red")) # 바꿔줄 수 있다.


A = " a b c d e f "

A = A.split(" ")
print(A)

A = ",".join(A) # ','로 토크나이징 했던 문자들을 다시 다 붙여준다.
print(A)

# .find() : 왼쪽부터 찾는
# .index() : 오른쪽부터 찾는

A = "korea japan usa"

print(A[A.find("japan"):])

print(A[A.index("japan"):])

# find와 index의 차이점
# 없는 문자 검색: index는 에러
# find는 -1이 반환

