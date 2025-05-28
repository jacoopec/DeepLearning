import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate synthetic 3D data
np.random.seed(42)
mean = [0, 0, 0]
cov = [[3, 1, 1], [1, 2, 1], [1, 1, 1]]  # Covariance matrix
X = np.random.multivariate_normal(mean, cov, 100)

# Step 2: Apply PCA to reduce from 3D to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get the directions of the two principal components
components = pca.components_
mean_vector = np.mean(X, axis=0)

# Step 3: Plotting
fig = plt.figure(figsize=(14, 6))

# 3D Plot: original data + PCA directions
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, label='Data')

# Add principal component vectors as arrows
for i in range(2):
    vec = components[i]
    ax1.quiver(
        mean_vector[0], mean_vector[1], mean_vector[2],
        vec[0], vec[1], vec[2],
        length=5, color='red' if i == 0 else 'green', linewidth=2,
        label=f'PC{i+1}'
    )

ax1.set_title('Original 3D Data with PCA Directions')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# 2D Plot: PCA projection
ax2 = fig.add_subplot(122)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='red', alpha=0.6)
ax2.set_title('2D Projection using PCA')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
