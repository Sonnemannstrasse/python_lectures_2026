import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

iris = sns.load_dataset("iris")


x = iris[iris['species'] != 'setosa'].iloc[:, 0:2].to_numpy()

y1 = np.full(50, 1)
y2 = np.full(50, 2)
y = np.ravel([y1, y2])

# 1. Выбирается класс модели
# 2. Выбираются гиперпараметры модели
model = RandomForestClassifier(n_estimators=100, max_depth=3)

# 3. На основе данных создается матрица признаков и целевой вектор

# 4. Обучение модели fit()
model.fit(x, y)

# 5. Обученная модель применяется к новым данным
#   5.1. Обучение с учителем - predict()
xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 100),
    np.linspace(x[:, 1].min() - 0.5, x[:, 1].max() + 0.5, 100),
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()
ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3], colors=['red', 'green'])

x_0 = iris[iris['species'] == 'versicolor'].iloc[:, 0].to_numpy()
y_0 = iris[iris['species'] == 'versicolor'].iloc[:, 1].to_numpy()
x_1 = iris[iris['species'] == 'virginica'].iloc[:, 0].to_numpy()
y_1 = iris[iris['species'] == 'virginica'].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5, label="versicolor")
plt.scatter(x_1, y_1, color="green", alpha=0.5, label="virginica")

plt.legend()
plt.show()
