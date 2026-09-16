import numpy as np
import matplotlib.pyplot as plt


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
