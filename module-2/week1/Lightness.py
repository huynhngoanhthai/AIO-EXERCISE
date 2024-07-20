import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('./content/dog.jpeg')


def rgb_to_lightness(pixel):
    return (max(pixel) + min(pixel)) / 2


# Tạo hình ảnh xám
gray_img_01 = np.zeros((img.shape[0], img.shape[1]))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        gray_img_01[i, j] = rgb_to_lightness(img[i, j])

print(gray_img_01[0, 0])

