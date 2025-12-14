import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load sample data (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data (PCA works best with scaled data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_reduced_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Reduced shape (PCA):", X_reduced_pca.shape)
print("Explained variance ratio of components:", pca.explained_variance_ratio_)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Target Class')
plt.show()
