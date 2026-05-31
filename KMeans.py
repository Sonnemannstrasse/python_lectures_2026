import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

iris = sns.load_dataset("iris")


x = iris[iris['species'] != 'setosa'].iloc[:, 0:2].to_numpy()

# 1. Выбирается класс модели
# 2. Выбираются гиперпараметры модели
model = KMeans(n_clusters=2)

# 3. На основе данных создается матрица признаков

# 4. Обучение модели fit()
model.fit(x)

# 5. Обученная модель применяется к новым данным
#   5.2. Обучение без учителя - predict()
xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 100),
    np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 100),
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()
ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['red', 'green'])

x_0 = iris[iris['species'] == 'versicolor'].iloc[:, 0].to_numpy()
y_0 = iris[iris['species'] == 'versicolor'].iloc[:, 1].to_numpy()
x_1 = iris[iris['species'] == 'virginica'].iloc[:, 0].to_numpy()
y_1 = iris[iris['species'] == 'virginica'].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5, label="versicolor")
plt.scatter(x_1, y_1, color="green", alpha=0.5, label="virginica")

centroids = model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=100, label="Центроиды")

plt.legend()
plt.show()
