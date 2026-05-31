import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

iris = sns.load_dataset("iris")


x = iris[iris['species'] != 'setosa'].iloc[:, 0:4].to_numpy()

y1 = np.full(50, 1)
y2 = np.full(50, 2)
y = np.ravel([y1, y2])

# 1. Выбирается класс модели
# 2. Выбираются гиперпараметры модели
model = PCA(n_components=2)

# 3. На основе данных создается матрица признаков и целевой вектор

# 4. Обучение модели fit()
model.fit(x)

# 5. Обученная модель применяется к новым данным
#   5.1. Обучение с учителем (нет метода predict(), поэтому transform())
x_pca = model.transform(x)

x_0 = x_pca[y == 1, 0]
y_0 = x_pca[y == 1, 1]
x_1 = x_pca[y == 2, 0]
y_1 = x_pca[y == 2, 1]

plt.scatter(x_0, y_0, color="red", alpha=0.5, label="versicolor")
plt.scatter(x_1, y_1, color="green", alpha=0.5, label="virginica")

plt.legend()
plt.show()
