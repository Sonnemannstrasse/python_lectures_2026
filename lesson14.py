import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x = np.linspace(-6, 6, 30)
y = np.linspace(-10, 10, 50)

print(x.shape)
print(y.shape)

X, Y = np.meshgrid(x, y)

print(X.shape)
print(Y.shape)

print(X, Y, sep='\n\n')

def f(x,y):
    return np.sin(np.sqrt(x**2 + y**2))

Z = f(X, Y)
print(Z.shape, end='\n\n')
print(Z)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.contour3D(X, Y, Z, 40, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(30, 30, 0) # Угол взгляда

plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.scatter(X, Y, Z, c=Z) # Подсвечиваем точки на одинаковом расстоянии

#ax.plot_wireframe(X, Y, Z)
ax.plot_surface(X, Y, Z, cmap='viridis')

plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')

angle = np.linspace(0, 1 * np.pi, 50)
r = np.linspace(0, 6, 30)

R, Angle = np.meshgrid(r, angle)

X = R * np.sin(angle[:, np.newaxis])
Y = R * np.cos(angle[:, np.newaxis])
Z = f(X, Y)

ax.plot_surface(X, Y, Z, cmap='viridis')
#ax.plot_trisurf(X, Y, Z, cmap='viridis')

plt.show()

# Триангуляция поверхности
fig = plt.figure()
ax = plt.axes(projection='3d')

angle = 1.5 * np.pi *  np.random.random(2000)
r = np.linspace(0, 6, 2000)

x = r * np.sin(angle)
y = r * np.cos(angle)
z = f(x, y)

ax.plot_trisurf(x, y, z, cmap='viridis')

plt.show()

# Seaborn - высокоуровневая надстройка к Matplotlib
# Matplotlib работает через numpy, а Seaborn через DataFrame

import seaborn as sns

sns.set_style('darkgrid')
cars = pd.read_csv('cars.csv.gz')
print(cars.head())

# Числовые данные
# парная диаграмма
sns.pairplot(data=cars, hue='transmission') # Он позволяет окрасить точки на всех графиках по значениям из указанного столбца

plt.show()

# Тепловая карта в Seaborn
'''
Тепловые карты (heatmaps) показывают плотность, интенсивность или распределение данных с помощью цвета. 
Проще говоря, они превращают скучные цифры в наглядную цветную картинку, где каждый оттенок имеет значение.
'''

cars_corr = cars[['year', 'selling_price', 'seats', 'mileage']]

sns.heatmap(cars_corr.corr(), cmap='viridis', annot=True)

plt.show()

# Диаграмма рассеяния
sns.scatterplot(x='year', y='selling_price', data=cars, hue='fuel')

plt.show()

# Диаграмма рассеяния + лин.регрессия
sns.relplot(x='seats', y='mileage', data=cars, kind='scatter', col='fuel', col_wrap=2, hue='transmission') 

plt.show()

# Диаграмма рассеяния + лин.регрессия
sns.lmplot(x='seats', y='mileage', data=cars, hue='fuel', col='transmission', col_wrap=2)

plt.show()

# Линейный график
sns.lineplot(x='seats', y='mileage', data=cars, hue='fuel')
plt.show()

# Сводная диаграмма
sns.jointplot(x='year', y='selling_price', data=cars)

plt.show()

# Сводная диаграмма
# sns.jointplot(x='year', y='selling_price', data=cars, kind='kde')
sns.jointplot(x='year', y='selling_price', data=cars,    kind='reg', joint_kws={'line_kws': {'color': 'purple', 'linewidth': 3, 'linestyle': '--'}})

plt.show()

sns.jointplot(x='year', y='selling_price', data=cars, hue='transmission')

# Категории и числа
sns.barplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission')

plt.show()

sns.catplot(x='fuel', y='selling_price', data=cars, hue='transmission', col='seller_type', col_wrap=2) # kind = 'bar' - will get the same graph like above one

plt.show()

sns.pointplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, hue='transmission')

sns.boxplot(x='fuel', y='selling_price', data=cars, hue='transmission')
'''
Выбросы (outliers) — точки за пределами усов, отображаются отдельными маркерами (обычно кружками).
'''
plt.show()

'''
Скрипка (violin) показывает распределение
'''
sns.violinplot(x='fuel', y='selling_price', data=cars, hue='transmission')

plt.show()

# Можно совмещать
fig, ax = plt.subplots()
sns.violinplot(x='fuel', y='selling_price', data=cars, hue='transmission', ax=ax)
sns.stripplot(x='fuel', y='selling_price', data=cars, hue='transmission', ax=ax, dodge=True)

plt.show()
