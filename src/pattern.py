import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

h, w, c = 720, 1280, 4
img_array = np.ones([h,w])*255
plt.imsave("white.bmp",img_array)
#image = Image.fromarray(img_array,'RGB')

#image.save("white.bmp", format='BMP')
print('printed')