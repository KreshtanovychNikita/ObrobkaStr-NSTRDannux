# Імпорт необхідних бібліотек
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Зчитування даних з Excel-файлу в DataFrame
df = pd.read_excel("PCOS_data_without_infertility.xlsx");

# Виведення перших рядків даних для огляду
df

# Виведення інформації про дані, такі як кількість рядків, стовпців і типи даних
df.info()

# Конвертація деяких стовпців у числовий тип даних з обробкою помилок
df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'], errors='coerce')
df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')

# Видалення зайвих стовпців
df.drop(['Unnamed: 44', 'FSH/LH', 'Waist:Hip Ratio', 'BMI'], axis=1, inplace=True)

# Повторний вивід інформації про дані після змін
df.info()

# Видалення рядків з пропущеними значеннями
df.dropna(inplace=True)

# Повторний вивід інформації про дані після видалення
df.info()

# Розрахунок кореляції між стовпцями і цільовою змінною PCOS
# та сортування їх за спаданням за абсолютним значенням кореляції
df.corr()['PCOS (Y/N)'].abs().sort_values(ascending=False)

# Створення нового DataFrame з важливими ознаками
df_new = df[['PCOS (Y/N)', 'Follicle No. (R)', 'Follicle No. (L)', 'Skin darkening (Y/N)', 'hair growth(Y/N)', 'Weight gain(Y/N)', 'Cycle(R/I)',
            'Fast food (Y/N)', 'Pimples(Y/N)', 'AMH(ng/mL)', 'Weight (Kg)']]

# Повторний вивід нового DataFrame
df_new

# Перетворення категоріального стовпця у числовий формат
df_new.loc[:, 'Fast food (Y/N)'] = df_new['Fast food (Y/N)'].astype(int)


# Виведення інформації про типи даних після конвертації
df_new.info()

# Розділення даних на навчальний та тестовий набори
X = df_new.drop(['PCOS (Y/N)'], axis=1)
y = df_new['PCOS (Y/N)']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Створення піднабору даних з PCOS (змінна PCOS (Y/N) = 1)
X_im = df_new[df_new['PCOS (Y/N)'] == 1]

# Побудова гістограм для ознак у піднаборі даних з PCOS
for column in X.columns:
    plt.hist(X_im[column], bins=10)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()
# Створення та навчання моделі логістичної регресії
logist = LogisticRegression(max_iter=1000)
logist.fit(x_train, y_train)

# Виведення точності моделі на навчальних та тестових даних
print(logist.score(x_train, y_train))
print(logist.score(x_test, y_test))

# Прогнозування значень на тестовому наборі та виведення графіку для порівняння з реальними значеннями
y_pred = logist.predict(x_test)
plt.figure()
plt.title('Predicted values vs real values in Test data')
plt.xlabel('Index')
plt.ylabel('PCOS')
plt.plot(y_test.index, y_test, label='Actual values')
plt.plot(y_test.index, y_pred, label='Predicted values')
plt.legend()
plt.show()

# Обчислення точності прогнозу за допомогою метрики accuracy_score
accuracy_score(y_test, y_pred)
print("End")
