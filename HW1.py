import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3)

# # make linear separable data
# points = 20
# dim = 2
# X1 = np.random.randn(points, dim) + np.array([1, 1])  # data1
# X2 = np.random.randn(points, dim) + np.array([-2, -2])  # data2
#
# # Label
# y1 = np.ones(points, dtype="uint8")
# y2 = np.zeros(points, dtype="uint8")
#
# X = np.concatenate((X1, X2))
# y = np.concatenate((y1, y2))
# # Add x0 = 1
# T = np.ones((points*2, 1), dtype="uint8")
# X = np.concatenate((T, X), axis=1)
# print(X[0])
# # Perceptron Learning Algorithm
# w = np.random.randn(1,dim+1)
# print(w)
# for iter in range(1000):
#     for i in range(len(X)):
#         if w@X[i] >= 0 and y[i]==0:
#             w = w - X[i]
#         elif w@X[i] <0 and y[i]==1:
#             w = w + X[i]
# print(w)
# plt.scatter(X1[:, 0], X1[:, 1], color='blue', label='Cụm 1 (y=1)')
# plt.scatter(X2[:, 0], X2[:, 1], color='red', label='Cụm 2 (y=0)')
#
# # Vẽ đường quyết định
# x_vals = np.linspace(-3, 3, 100)  # Các giá trị x
# # Phương trình đường thẳng: w0 + w1*x + w2*y = 0 => y = -(w0 + w1*x) / w2
# y_vals = -(w[0][0] + w[0][1] * x_vals) / w[0][2]
# plt.plot(x_vals, y_vals, color='green', label='Đường quyết định (w)')
#
# # Cấu hình biểu đồ
# plt.axhline(0, color='black',linewidth=0.5, linestyle='--')
# plt.axvline(0, color='black',linewidth=0.5, linestyle='--')
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Phân cụm dữ liệu và đường quyết định của Perceptron')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# data 3 dim:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tạo dữ liệu tuyến tính phân tách
np.random.seed(42)
n_samples = 50

# Tạo nhóm 1 (class 0)
x1 = np.random.rand(n_samples)
y1 = np.random.rand(n_samples)
z1 = 0.5 * x1 + 0.3 * y1 + 0.2 + np.random.normal(0, 0.02, n_samples)  # Mặt phẳng z = 0.5x + 0.3y + 0.2

# Tạo nhóm 2 (class 1)
x2 = np.random.rand(n_samples)
y2 = np.random.rand(n_samples)
z2 = 0.5 * x2 + 0.3 * y2 + 0.5 + np.random.normal(0, 0.02, n_samples)  # Dịch lên để đảm bảo phân tách

# Vẽ các điểm dữ liệu
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, y1, z1, c='blue', label='Class 0')
ax.scatter(x2, y2, z2, c='red', label='Class 1')

# Vẽ mặt phẳng tuyến tính phân tách
X_plane, Y_plane = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
Z_plane = 0.5 * X_plane + 0.3 * Y_plane + 0.35  # Mặt phẳng tuyến tính

ax.plot_surface(X_plane, Y_plane, Z_plane, color='green', alpha=0.5)

# Thiết lập nhãn trục
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.title("Linearly Separable Data in 3D")
ax.legend()

plt.show()
