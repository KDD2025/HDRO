import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
# data cluster and domain_ratio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.preprocessing import normalize

data_path = '/home/zhangyabin/Multi-scenarios-Robust/train_val_test_split/douban'
df_train = pd.read_csv(data_path + '/train_dataset.csv')
# df_val = pd.read_csv(data_path + '/val_dataset.csv')
# df_test = pd.read_csv(data_path + '/test_dataset_0.8_0.8.csv')
del df_train["label"]
# del df_val["label"]
# del df_test["label"]
df_train=df_train[:100]
# df_val = df_val[:100]
#df_test = df_test[:100]
data = df_train

# # 生成一些未知的多维数据，这里以二维正态分布为例
# np.random.seed(42)
# unknown_data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000)
numeric_columns = data.select_dtypes(include=[np.number]).columns
user_item_numeric = data[numeric_columns]

# 使用 PCA 进行降维
pca = PCA(n_components=2)
embedding = pca.fit_transform(user_item_numeric)

# 将用户和物品嵌入连接，形成样本表示
samples_representation = np.hstack((embedding[:, 0].reshape(-1, 1), embedding[:, 1].reshape(-1, 1)))

data = samples_representation
# 使用 K-Means 聚类将数据分成3类
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(data)

# 统计每个类别的样本个数
counts_by_cluster = np.bincount(labels)

# 计算正则化后的类别分布
normalized_distribution = normalize(counts_by_cluster.reshape(1, -1), norm='l1')
print(f'normalized_distribution11',normalized_distribution[0])
# 打印每个类别的样本个数和正则化分布
for cluster, count, normalized_prob in zip(range(n_clusters), counts_by_cluster, normalized_distribution[0]):
    print(f'Cluster {cluster + 1} - Sample Count: {count}, Normalized Probability: {normalized_prob:.2f}')


