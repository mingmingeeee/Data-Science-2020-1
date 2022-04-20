
# Matplotlib

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

print(x)
print(y)

plt.plot(x, y)
#       x축 y축

y_sin = np.sin(x)
y_cos = np.cos(y)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
# 라벨 붙이기
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine','Cosine'])


# subplot -> 그림 하나 안에 차트 여러개

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('Sine')
# 새로로는 두개 가로로는 한개
# 그리려고 하는 것은 그 중에 1번째이다.

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()

