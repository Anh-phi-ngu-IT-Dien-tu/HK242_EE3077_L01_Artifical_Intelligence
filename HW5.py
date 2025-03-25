import cv2
import numpy as np
from scipy.spatial.distance import cdist
import math

np.random.seed(3)
img_path = "C:/Users/DELL/Desktop/test.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (640,640))
height, width, _ = img.shape
pixel_data = img.reshape(-1, 3)
original_bits = height * width * 3 * 8

K = 16
centers = pixel_data[np.random.choice(pixel_data.shape[0],K, replace=False)]
labels = np.random.randint(0, K, len(pixel_data))
iter = 0
while True:
    iter += 1
    distances = cdist(pixel_data, centers)
    nearest_center = np.argmin(distances, axis=1)
    if not np.array_equal(labels, nearest_center):
        labels = nearest_center
    else:
        break
    new_centers = []
    for k in range(K):
        same_label_index = np.where(labels == k)[0]
        if len(same_label_index > 0):
            mean_point = [pixel_data[index] for index in same_label_index]
        else:
            mean_point = centers[k]
        new_centers.append(np.mean(mean_point, axis=0))
    centers = np.array(new_centers)

compressed_img = centers[labels].reshape(height, width, 3).astype(np.uint8)
compressed_bits = (height * width * math.log2(K)) + (K * 24)

print(f"Số lần lặp: {iter} lần")
print(f"Số bit của ảnh gốc: {original_bits} bits")
print(f"Số bit của ảnh đã nén: {compressed_bits} bits")

puzzle_img = cv2.hconcat([img,compressed_img])
cv2.imshow("Image Compression", puzzle_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




