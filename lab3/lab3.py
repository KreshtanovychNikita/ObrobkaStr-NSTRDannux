# Імпортуємо необхідні бібліотеки
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
import joblib

# Зчитуємо дані з CSV-файлу в DataFrame
df = pd.read_csv('wine-clustering.csv')

# Виводимо інформацію про дані, такі як кількість рядків, стовпців і типи даних
df.info()

# Створюємо об'єкт PCA з двома компонентами
pca = PCA(2)

# Виконуємо проекцію даних на два головні компоненти (PCA)
projection = pd.DataFrame(columns=['pca1','pca2'], data=pca.fit_transform(df))

# Виводимо отриману проекцію даних
print(projection)

# Створюємо модель кластеризації KMeans з 3 кластерами
kmeans = KMeans(n_clusters=3, n_init=10)


# Навчаємо модель KMeans на проекції даних
kmeans.fit(projection)

# Прогнозуємо приналежність кластерів для кожного екземпляра
projection['cluster'] = kmeans.predict(projection)

# Знаходимо центроїди кластерів
centroids = kmeans.cluster_centers_

# Визначаємо координати центроїдів для графіку
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Створюємо графік розсіювання з кластерами та центроїдами
plt.figure(figsize=(6,6))
sns.scatterplot(data=projection, x='pca1', y='pca2', hue='cluster', palette="deep")
sns.scatterplot(x=centroids_x, y=centroids_y, marker='o', c=['black'])
plt.title("Clustering Plot")
plt.xlabel('pca1 Data')
plt.ylabel('pca2 Data')
plt.legend()
plt.show()

# Розраховуємо середнє значення для кожної колонки
means = df.mean()

# Побудова кругової діаграми
plt.figure(figsize=(8, 8))
plt.pie(means, labels=means.index, autopct='%1.1f%%', startangle=140)
plt.title('Середнє значення для кожного атрибута вина')
plt.axis('equal')  # Забезпечує круглу форму діаграми
plt.show()

# Обчислюємо коефіцієнт силуету для оцінки якості кластеризації
metrics.silhouette_score(projection[['pca1', 'pca2']], projection['cluster'])

# Зберігаємо модель KMeans у файлі
joblib.dump(kmeans, 'kmeans.pkl')

print(kmeans)
