import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC"
]

plt.rcParams["axes.unicode_minus"] = False

def kmeans(X, K, max_iter=100, tol=1e-4):

    # =========================
    # 1. 初始化中心
    # =========================

    indices = np.random.choice(len(X), K, replace=False)

    centroids = X[indices].copy()

    # =========================
    # 2. 开始迭代
    # =========================

    for i in range(max_iter):

        # -------------------------
        # Step 1：计算距离
        # -------------------------

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # -------------------------
        # Step 2：分配类别
        # -------------------------

        labels = np.argmin(distances, axis=1)

        # -------------------------
        # Step 3：保存旧中心
        # -------------------------

        old_centroids = centroids.copy()

        # -------------------------
        # Step 4：更新中心
        # -------------------------

        for k in range(K):

            cluster_points = X[labels == k]

            if len(cluster_points) > 0:

                centroids[k] = cluster_points.mean(axis=0)

        # -------------------------
        # Step 5：判断收敛
        # -------------------------

        movement = np.linalg.norm(centroids - old_centroids)

        if movement < tol:
            print(f"K-Means 在第 {i + 1} 次迭代后收敛")
            break

    return labels, centroids


# ==================================
# 构造数据
# ==================================

np.random.seed(42)

X1 = np.random.randn(100, 2) + [2, 2]
X2 = np.random.randn(100, 2) + [-2, -2]
X3 = np.random.randn(100, 2) + [5, -3]

X = np.vstack([X1, X2, X3])


# ==================================
# K-Means
# ==================================

labels, centroids = kmeans(X, K=3)


# ==================================
# 可视化
# ==================================

plt.scatter(X[:, 0], X[:, 1], c=labels)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200)

plt.show()

##使用API
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



# 使用 make_blobs 生成 3 个簇，每个簇 100 个点
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2)

fig, ax = plt.subplots(2, figsize=(8, 8))
ax[0].scatter(X[:, 0], X[:, 1], s=50, c="gray", label="原始数据")
ax[0].set_title("原始数据")
ax[0].legend()

# 使用 K-Means 聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X) # 预测每个点的簇标签
centers = kmeans.cluster_centers_ # 获取簇中心

ax[1].scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans)
ax[1].scatter(centers[:, 0], centers[:, 1], s=200, c="red", marker="o", label="簇中心")
ax[1].set_title("K-means 聚类结果 (K=3)")
ax[1].legend()
plt.show()