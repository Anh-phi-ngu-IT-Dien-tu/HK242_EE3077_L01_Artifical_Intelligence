import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3)

# make linear separable data
points = 20
dim = 3
X1 = np.random.randn(points, dim) + np.array([1, 1, 1])  # data1
X2 = np.random.randn(points, dim) + np.array([-2, -2, -2])  # data2

# Label
y1 = np.ones(points, dtype="uint8")
y2 = np.zeros(points, dtype="uint8")

X = np.concatenate((X1, X2))
y = np.concatenate((y1, y2))
# Add x0 = 1
T = np.ones((points*2, 1), dtype="uint8")
X = np.concatenate((T, X), axis=1)
print(f"X[0]:{X[0]}")

# Perceptron Learning Algorithm
w = np.random.randn(1,dim+1)
print(f"w init:{w}")
for iter in range(1000):
    for i in range(len(X)):
        if w@X[i] >= 0 and y[i]==0:
            w = w - X[i]
        elif w@X[i] <0 and y[i]==1:
            w = w + X[i]
print(f"w perceptron: {w}")
# simulate
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], color='blue', label='Data 1 (y=1)')
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], color='red', label='Data 2 (y=0)')

x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
x_vals, y_vals = np.meshgrid(x_vals, y_vals)
z_vals = -(w[0][0] + w[0][1] * x_vals + w[0][2] * y_vals) / w[0][3]
ax.plot_surface(x_vals, y_vals, z_vals, color='green', alpha=0.5)

ax.legend()
ax.view_init(elev=30, azim=30)
plt.show()