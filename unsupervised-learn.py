# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset (Study Hours, Attendance)
data = np.array([
    [1, 40], [2, 50], [3, 55], [4, 70], [5, 75],
    [6, 80], [7, 85], [8, 90], [9, 95]
])

#Clustering == grouping (cluster --> how many ??)

# Apply K-Means Clustering (2 clusters: low performers, high performers)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Get cluster labels
labels = kmeans.labels_

# Plot the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=100)
plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.title("Student Clusters")
plt.show()
