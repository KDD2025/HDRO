import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 假设 user_item_matrix 是用户-物品评分矩阵，每行对应一个用户，每列对应一个物品
user_item_matrix = np.array([[1, 2, 0],
                             [0, 3, 4],
                             [5, 0, 6],
                             [7, 8, 0]])

# 使用 PCA 进行降维
pca = PCA(n_components=2)
embedding = pca.fit_transform(user_item_matrix)

# 将用户和物品嵌入连接，形成样本表示
samples_representation = np.hstack((embedding[:, 0].reshape(-1, 1), embedding[:, 1].reshape(-1, 1)))

# 使用 k-means 进行聚类，这里假设分为 2 类
kmeans = KMeans(n_clusters=2)
kmeans.fit(samples_representation)

# 获取聚类结果
labels = kmeans.labels_

# 将每个样本属于的类别添加到原始样本后
user_item_with_labels = np.hstack((user_item_matrix, labels.reshape(-1, 1)))

# 输出新的样本
print("原始用户-物品评分矩阵及其所属类别：")
print(user_item_with_labels)
