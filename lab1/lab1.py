import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Завантаження даних
df = pd.read_csv('SBAcase.11.13.17.csv', delimiter=';')

# Визначення числових та категорійних стовпців
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('Default')  # Виключаємо цільову змінну

categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Трансформація для числових фічів: заповнення пропущених значень та нормалізація
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Трансформація для категорійних фічів: заповнення пропущених значень та однокове кодування
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Перетворення даних за допомогою ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Створення пайплайну, який включає перетворення та модель регресії
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Визначення фічів та цільової змінної
X = df.drop('Default', axis=1)
y = df['Default']

# Розділення даних на навчальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Навчання пайплайну
pipeline.fit(X_train, y_train)

# Прогнозування
predictions = pipeline.predict(X_test)

# Оцінювання моделі
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
