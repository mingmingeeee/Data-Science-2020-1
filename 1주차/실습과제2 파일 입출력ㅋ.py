# 실습과제 2
# comma seperate value

f = open('###', 'r')
head = f.readline() # 파일의 첫 라인을 읽는다.
print(head)
data = f.readlines()  # 파일의 나머지 모든 라인을 읽어온다. 라인들은 리스트로 저장된다.
f.close()

print(len(data))
print(type(data[0]), data[0])