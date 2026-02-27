# K-MEANS CLUSTERING FOR CUSTOMER SEGMENTATION

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())

# 3. Select Relevant Features (Purchasing Pattern)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
df['Cluster'] = clusters

# 6. Get Centroids
centroids = kmeans.cluster_centers_

# =========================================================
# 7. Plot Clusters in 2D Space (Scaled Features)
# =========================================================

plt.figure(figsize=(8,6))

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='viridis', s=50)
# alternative of 'viridis' are 'plasma', 'cool', 'rainbow'

# Plot centroids
plt.scatter(centroids[:,0], centroids[:,1], 
            c='red', s=200, marker='X')

plt.xlabel("Scaled Annual Income")
plt.ylabel("Scaled Spending Score")
plt.title("K-Means Clustering of Customers")
plt.show()

# =========================================================
# 8. Cluster Analysis
# =========================================================

print("\nCluster-wise Mean Values:")
cluster_analysis = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_analysis)

# Optional: Count customers in each cluster
print("\nNumber of customers in each cluster:")
print(df['Cluster'].value_counts())

# =========================================================
# 9. Interpret Clusters Automatically
# =========================================================

for i in range(5):
    income = cluster_analysis.iloc[i][0]
    spending = cluster_analysis.iloc[i][1]
    
    print(f"\nCluster {i} Analysis:")
    
    if income > 60 and spending > 60:
        print("-> High Income, High Spending (Premium Customers)")
    elif income < 40 and spending < 40:
        print("-> Low Income, Low Spending (Budget Customers)")
    elif income > 60 and spending < 40:
        print("-> High Income, Low Spending (Conservative Customers)")
    elif income < 40 and spending > 60:
        print("-> Low Income, High Spending (Impulsive Buyers)")
    else:
        print("-> Medium Income & Medium Spending (Average Customers)")
