# =========================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# =========================================================

# 1. Import required libraries
import numpy as np                   # numerical operations
import pandas as pd                  # data handling
import matplotlib.pyplot as plt      # plotting
from sklearn.datasets import load_wine   # wine dataset
from sklearn.preprocessing import StandardScaler  # scaling
from sklearn.decomposition import PCA  # PCA

# 2. Load wine dataset
wine = load_wine()

X = wine.data          # feature data
y = wine.target        # target classes

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 5. Explained variance ratio
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# 6. Plot cumulative variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("PCA Variance Explained")
plt.show()

# 7. 2D PCA (first two components)
pca_2 = PCA(n_components=2)
X_2D = pca_2.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_2D[:,0], X_2D[:,1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Projection")
plt.show()

# 8. 3D PCA (first three components)
from mpl_toolkits.mplot3d import Axes3D

pca_3 = PCA(n_components=3)
X_3D = pca_3.fit_transform(X_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_3D[:,0], X_3D[:,1], X_3D[:,2], c=y)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA 3D Projection")

plt.show()
