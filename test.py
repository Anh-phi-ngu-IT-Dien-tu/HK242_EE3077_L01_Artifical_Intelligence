import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# data gồm đường kính trên và dưới
X = np.array([
    [5, 6], [4, 4.7], [6, 6.3], [5.7, 6.1],   # quả quýt
    [7, 8], [6.7, 7.5], [6, 6.5], [5, 5.7],   # quả ổi
    [3.4, 1], [3, 1.5], [3, 1.8], [2.8, 1.6]  # quả táo xanh
])

# chiều cao
y = np.array([
    5.8, 5.0, 5.4, 5.2,
    6.5, 6.2, 5.6, 6.3,
    4.5, 4.7, 4.4, 4.3 ])

# mô hình LN
model = LinearRegression()
model.fit(X, y)
#trọng số w và bias
w = model.coef_
b = model.intercept_
print("w =", w)
print("b =", b)

# Dự đoán chiều cao quả bằng mô hình trên
new_sample = np.array([[5.5, 6.0]])
predicted_height = model.predict(new_sample)
print("Chiều cao của quả:", predicted_height[0])

#Simulate
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data points')

x_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
y_vals = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_grid = w[0] * X_grid + w[1] * Y_grid + b

ax.plot_surface(X_grid, Y_grid, Z_grid, color='red', alpha=0.5)

ax.set_xlabel('Đường kính trên')
ax.set_ylabel('Đường kính dưới')
ax.set_zlabel('Chiều cao')
ax.legend()
plt.show()