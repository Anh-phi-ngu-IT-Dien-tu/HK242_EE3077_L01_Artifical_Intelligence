# 3 class (min)
# dài, rộng, cân nặng,... (min 3) feature
# a) áp dụng các pp -> classification
# b) dùng KNN
# HW3: tìm data giải cho bài toán linear regression

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Process data from excel
data_path = "data.xlsx"
df = pd.read_excel(data_path)
data = df.iloc[0:10, 2:6]
dataset = data.astype(float).T.values.tolist()
# print(dataset)
table = [[row[i:i+3] for i in range(0, len(row), 3)] for row in dataset]
class_0 = [item[0] for item in table]
class_1 = [item[1] for item in table]
class_2 = [item[2] for item in table]
# print(f"class_0:{class_0}")
# print(f"class_1:{class_1}")
# print(f"class_2:{class_2}")
X = class_0 + class_1 + class_2
# Label
y = [0]*len(class_0) + [1]*len(class_1) + [2]*len(class_2)

X_array = np.array(X)
y_array = np.array(y)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_array[:, :2], y_array)

h = 0.1
x_min, x_max = X_array[:, 0].min() - 1, X_array[:, 0].max() + 1
y_min, y_max = X_array[:, 1].min() - 1, X_array[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)


plt.scatter(X_array[:len(class_0), 0], X_array[:len(class_0), 1], color='red', label='Class 0')
plt.scatter(X_array[len(class_0):len(class_0)+len(class_1), 0],
            X_array[len(class_0):len(class_0)+len(class_1), 1], color='blue', label='Class 1')
plt.scatter(X_array[-len(class_2):, 0], X_array[-len(class_2):, 1], color='green', label='Class 2')
plt.legend()
plt.grid(True)
plt.show()