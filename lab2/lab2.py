# Імпорт необхідних бібліотек
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Зчитування даних з CSV-файлу в DataFrame
df = pd.read_csv("water_potability.csv")

# Виведення інформації про дані, такі як кількість рядків, стовпців і типи даних
df.info()

# Видалення рядків з пропущеними значеннями
df.dropna(inplace=True)

# Виведення інформації про дані після видалення пропущених значень
df.info()

# Розрахунок кореляції між стовпцями і цільовою змінною 'Potability'
correlations = df.corr()['Potability'].abs().sort_values(ascending=False)

# Виведення кореляцій у консоль
print("Кореляції з Potability:")
print(correlations)

# Побудова графіка кореляцій
plt.figure(figsize=(10, 6))
correlations.plot(kind='bar', color='skyblue')
plt.title('Кореляції з Potability для кожного стовпця')
plt.xlabel('Стовпець')
plt.ylabel('Абсолютна кореляція')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Розділення даних на ознаки (X) та цільову змінну (y)
X = df.drop(['Potability'], axis=1)
y = df['Potability']

# Розділення даних на навчальний та тестовий набори у співвідношенні 80% до 20%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Створення моделі класифікації Random Forest
rfc = RandomForestClassifier()

# Навчання моделі на навчальних даних
rfc.fit(x_train, y_train)

# Прогнозування значень на тестовому наборі
y_pred = rfc.predict(x_test)

# Виведення точності моделі на тестових даних
print("Accuracy = ", accuracy_score(y_test, y_pred))

# Побудова графіку для порівняння реальних і прогнозованих значень цільової змінної
plt.figure()
plt.title('Predicted values vs real values in Test data')
plt.xlabel('Index')
plt.ylabel('Potability')
plt.plot(y_test.index, y_test, label='Actual values')
plt.plot(y_test.index, y_pred, label='Predicted values')
plt.legend()

# Відображення графіка
plt.show()
