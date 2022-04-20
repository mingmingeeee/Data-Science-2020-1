import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

img = imread("c:\\cat.jpg")

print(type(img))
print(img.ndim)
print(img.shape)
# 높이 픽셀 R,G,B

img_tinted = img * [1, 0.95, 0.9]

# 첫번째 subplot에는 원본
plt.subplot(1, 2, 1)
plt.imshow(img)

# 두번째 subplot에는 색변화된 이미지
plt.subplot(1, 2, 2)

plt.imshow(np.uint8(img_tinted))

plt.show() # display 해주기 위함