import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load and prepare data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(iris.target, columns=['Targets'])

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Plot K-Means results
colormap = np.array(['red', 'lime', 'black'])
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[kmeans_labels], s=40)
plt.title('K-Means Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
print('K-Means Accuracy Score:', metrics.accuracy_score(y, kmeans_labels))
print('K-Means Confusion Matrix:\n', metrics.confusion_matrix(y, kmeans_labels))

# Gaussian Mixture Model clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3)
gmm_labels = gmm.fit_predict(X_scaled)

# Plot GMM results
plt.subplot(2, 2, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_labels], s=40)
plt.title('GMM Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

print('GMM Accuracy Score:', metrics.accuracy_score(y, gmm_labels))
print('GMM Confusion Matrix:\n', metrics.confusion_matrix(y, gmm_labels))

plt.tight_layout()
plt.show()
