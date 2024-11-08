# K-近邻算法（K-Nearest Neighbors, KNN）

## 0x01. KNN概念

K-近邻算法（K-Nearest Neighbors, 简称 KNN）是一种**基于实例的学习算法**，用于分类和回归问题。KNN 的核心思想是，通过计算样本之间的距离，将新样本归类到距离最近的 \( K \) 个邻居所属的类别中。

KNN 被称为懒惰学习算法，因为在训练阶段并没有显式构建模型，而是把所有训练数据存储起来，等到需要分类或预测时才进行计算。

## 0x02. KNN的工作原理

1. **选择距离度量方式**：通常使用**欧氏距离**来度量样本间的距离：
   $
   d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} 
   $
   其中 \( x_i \) 和 \( y_i \) 分别是样本 \( x \) 和样本 \( y \) 在第 \( i \) 个特征上的取值。

2. **确定最近的 \( K \) 个邻居**：计算待分类样本与训练集中所有样本之间的距离，选出距离最近的 \( K \) 个样本。

3. **分类或回归**：
   - **分类问题**：选择 \( K \) 个邻居中出现次数最多的类别作为新样本的类别。
   - **回归问题**：取 \( K \) 个邻居的目标值的平均值作为新样本的预测值。

## 0x03. KNN的参数

1. **K值**：K 值是 KNN 的关键参数，表示参与决策的邻居数量。选择合适的 \( K \) 值对于算法性能至关重要。
   - **小 K 值**：较小的 \( K \) 值会使模型对噪声数据更敏感，容易导致过拟合。
   - **大 K 值**：较大的 \( K \) 值会使模型更加平滑，但可能会忽略局部结构，导致欠拟合。

2. **距离度量方式**：
   - **欧氏距离（Euclidean Distance）**：常用于连续数据，公式如前所述。
   - **曼哈顿距离（Manhattan Distance）**：常用于离散特征：
     $$ 
     d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
     $$
   - **闵可夫斯基距离（Minkowski Distance）**：可以视为欧氏距离和曼哈顿距离的泛化形式：
     $$ 
     d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}} 
     $$
     当 \( p = 2 \) 时为欧氏距离，\( p = 1 \) 时为曼哈顿距离。

3. **加权 KNN**：在 KNN 中可以给邻居的距离加权，距离越近的邻居权重越大，使得预测更加准确。


## 0x04. 优缺点

### 优点：
- **简单直观**：KNN 是一种非常直观的算法，简单易实现，不需要训练过程。
- **适应性强**：KNN 不依赖特定的数据分布，适用于分类和回归任务。

### 缺点：
- **计算量大**：KNN 在预测时需要计算所有训练样本的距离，计算复杂度较高，特别是在数据量较大的情况下。
- **高维数据问题**：KNN 在高维空间中会遭遇“维度灾难”问题，距离度量变得不再可靠。
- **对不平衡数据敏感**：在类别不平衡的数据集中，KNN 倾向于将样本归类到多数类。

## 0x05. KNN的应用场景

1. **文本分类**：KNN 可以用于自然语言处理中的文本分类任务。通过将文本转换为向量表示，计算不同文本之间的相似度，从而实现分类。
2. **推荐系统**：KNN 可以通过寻找相似用户或相似物品，来为用户提供推荐。
3. **图像分类**：KNN 也可以用于图像分类，尤其是基于低维特征表示的图像识别任务。

## 0x06. KNN的实现

下面是一个基于 Python 的 KNN 实现的代码示例：

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建KNN分类器，选择K=3
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率为: {accuracy}")
```

## 0x07. KNN优化
KD-树和球树：为了加速距离计算，可以使用 KD-树（k-d tree）或球树（Ball Tree）等数据结构来存储训练样本，加快 KNN 的查询效率。
标准化和归一化：在 KNN 中，距离度量对特征的量纲敏感，因此常常需要对数据进行标准化或归一化处理，以消除不同特征的量纲差异对结果的影响。
降维：对于高维数据，可以使用主成分分析（PCA）等降维方法来降低特征维度，减小计算量并提高性能。

# K-Means 聚类算法

## 0x01. K-Means 概念

**K-Means** 是一种**无监督学习算法**，用于将数据集划分为 **K** 个互不重叠的子集（即簇，clusters）。该算法试图最小化每个簇内样本点与簇中心的距离平方和。K-Means 是最常见的聚类算法之一，适用于高效地处理大规模数据。

### K-Means 算法的核心思想：
1. 给定要划分的数据集和一个参数 **K**（簇的数量），算法的目标是将数据集划分为 **K** 个簇，使得每个簇中的数据点到其簇中心的距离平方和最小化。
2. **簇中心（centroid）** 是指在每个簇中所有数据点的几何中心。

## 0x02. K-Means 的工作流程

K-Means 算法的具体步骤如下：

1. **初始化**：
   - 随机选择 **K** 个初始的簇中心（centroid），这可以是从数据集中随机选取 K 个点作为初始中心，也可以采用更复杂的初始化策略，如 K-Means++。
   
2. **分配样本到最近的簇中心**：
   - 对于每个数据点，计算它到所有簇中心的欧氏距离，将其分配给距离最近的簇。

3. **更新簇中心**：
   - 根据新的簇划分，计算每个簇中所有数据点的均值，将均值作为新的簇中心。

4. **重复步骤 2 和 3**：
   - 反复执行分配和更新步骤，直到簇中心不再发生明显变化或达到预设的迭代次数。

### 伪代码：

```text
Input: 数据集 D，簇的个数 K
Output: K 个簇及其簇中心

1. 随机选择 K 个初始簇中心
2. Repeat:
     a. 对于每个数据点，计算其与所有簇中心的距离，并将其分配到最近的簇
     b. 更新每个簇的簇中心，簇中心为该簇中所有数据点的均值
3. 直到簇中心不再变化或达到最大迭代次数
```
## 0x03. K-Means 中的距离度量

**欧氏距离（Euclidean Distance）** 是 K-Means 中常用的距离度量方式，计算公式为：
\[
d(x, c) = \sqrt{\sum_{i=1}^{n} (x_i - c_i)^2}
\]
其中：
- \( x \) 是数据点，
- \( c \) 是簇中心，
- \( n \) 是特征的维度。

## 0x04. K-Means 的初始化方法

### 1. 随机初始化：
最简单的方法是从数据集中随机选择 K 个点作为初始簇中心。

### 2. K-Means++ 初始化：
K-Means++ 是 K-Means 的一种改进初始化方法，目的是使初始簇中心更加分散，从而提高聚类质量。K-Means++ 的步骤如下：
1. 首先随机选择一个数据点作为第一个簇中心。
2. 对于剩余的数据点，选择到现有簇中心最远的点作为下一个簇中心。这样可以保证每个新的簇中心尽量远离现有的簇中心。
3. 重复上述步骤，直到选出 K 个簇中心。

## 0x05. K-Means 的优缺点

### 优点：
1. **简单高效**：K-Means 算法非常简单，易于实现，且计算复杂度较低，适合处理大规模数据。
2. **收敛速度快**：通常在少量迭代后就能收敛。
3. **线性复杂度**：时间复杂度为 \(O(n \cdot k \cdot t)\)，其中 \(n\) 是样本数，\(k\) 是簇的数量，\(t\) 是迭代次数。

### 缺点：
1. **K 值敏感**：K 值的选择对结果影响很大，且没有确定的方法来直接确定最佳的 K 值。
2. **初始值敏感**：算法对初始簇中心敏感，不同的初始化可能导致不同的聚类结果，K-Means++ 能缓解这个问题。
3. **局限于球形簇**：K-Means 假设每个簇是球形的，适合形状规则、大小相似的簇。对于复杂形状的簇，K-Means 表现不佳。
4. **容易陷入局部最优**：由于是基于距离的优化，K-Means 可能陷入局部最优解。

## 0x06. K 值的选择

### 肘部法（Elbow Method）：
肘部法是选择 K 值的一种常用方法。具体做法是：绘制不同 K 值下聚类结果的代价函数图（通常为簇内距离平方和之和），找到拐点位置对应的 K 值，这个拐点就像“肘部”一样。

1. 计算每个 K 值下的 SSE（Sum of Squared Errors，误差平方和）：
\[
SSE = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
\]
其中 \( C_i \) 是第 \(i\) 个簇，\( \mu_i \) 是其簇中心。

2. 随着 K 值增加，SSE 会逐渐减小，但当 K 超过某个值时，SSE 减少的幅度会变得很小。此时的 K 就是合适的 K 值。

### 示例图（Elbow Method）：

在上图中，当 K = 3 时，图像出现拐点，这是最佳的 K 值。

## 0x07. K-Means 的代码实现

以下是使用 Python 实现 K-Means 算法的代码示例，使用了 `scikit-learn` 库：

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成样本数据
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=0.60)

# 创建KMeans模型并拟合数据
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering')
plt.show()
在此代码中，我们使用了 make_blobs 函数生成了 300 个样本数据，并使用 KMeans 模型对数据进行聚类，最终通过 matplotlib 库将结果进行可视化。

0x08. 总结
K-Means 是一种经典的聚类算法，具有简单高效的优点，但在选择簇的数量（K 值）和初始簇中心时需要特别注意。通过使用肘部法（Elbow Method）或其他优化方法，可以帮助选择合适的 K 值。同时，K-Means++ 初始化算法可以显著提升聚类结果的稳定性和质量。
