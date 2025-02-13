import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import timeit

# Чтение данных
file_path = 'yearssalary.xlsx'
data = pd.read_excel(file_path)

# Определение входных данных и целевой переменной
X = data[['YearsExperience']].values
y = data['Salary'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели SVR
model = SVR(kernel='linear')

# Настройка гиперпараметров
parameters = {
    'C': [0.001, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
    'epsilon': [0.001, 0.01, 0.1, 0.5, 1, 2]
}

grid_search = GridSearchCV(model, parameters, cv=15, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

# Лучшая модель
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Обучение модели с лучшими параметрами с использованием timeit
start_time = timeit.default_timer()

best_model.fit(X_train, y_train)

end_time = timeit.default_timer()
training_time = end_time - start_time

# Предсказание результата
predictions = best_model.predict(X_test)

# Вычисление MAE, MSE и оценка точности модели
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("Model accuracy:", r2_score(y_test, predictions))
print("Среднее время обучения:", training_time, "секунд")

