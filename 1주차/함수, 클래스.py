# 함수 (Fuction)

def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

def hello(name, loud = False): # loud를 안주면 default값인 false가 되어버림.
    if loud:
        print('Hello, {}'.format((name.upper())))
    else:
        print('Hello, {}'.format(name))

hello('Bob')
hello('Fred', loud = True)

# 클래스

class Greeter: # 이름은 대문자

    # Constructor - 생성자
    def __init__(self, name):
        self.name = name # name이라는 것은... 필드이다
    # self.name :: 생성자 뿐만 아닌 다른 메서드에서도 사용 가능

    # Instance method - 메서드
    def greet(self, loud = False): # 매개변수
        # self 로 시작되면 object 멤버
        # 아니면 class, static 멤버
        if loud:
            print('Hello, {}'.format((self.name.upper())))
        else:
            print('Hello, {}'.format(self.name))

g = Greeter('Fred') # self는 생략
g.greet()
g.greet(loud = True)