# nén nhiều ảnh dùng K means clustering
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Đọc ảnh
image = cv2.imread("1.jpg")
print(image)
print(type(image))
