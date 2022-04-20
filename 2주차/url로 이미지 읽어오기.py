from PIL import Image
from urllib.request import urlopen

# url = 'https://newevolutiondesigns.com/images/freebies/colorful-background-14.jpg'
url = 'https://github.com/ohheum/DS2020/blob/master/assets/cat.jpg?raw=true'
img = Image.open(urlopen(url))
plt.imshow(img)
plt.show()