import cv2
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang ảnh nhị phân
img = cv2.imread('sample/data/0000_samples.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Chuyển sang dạng boolean để dùng với skeletonize
binary_bool = binary.astype(bool)

# Áp dụng skeletonization
skeleton = skeletonize(binary_bool)

# Hiển thị ảnh kết quả
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(binary, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Skeletonized')
plt.imshow(skeleton, cmap='gray')
plt.show()
