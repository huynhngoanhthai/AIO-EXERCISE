import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('./content/dog.jpeg')

gray_img_average = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3

# In giá trị điểm ảnh đầu tiên
print(gray_img_average[0, 0])
