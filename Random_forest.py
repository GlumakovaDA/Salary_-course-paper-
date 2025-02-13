import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import timeit

# Загрузка данных
file_path = 'C:\\Users\\USER\\Desktop\\курсовая_работа2сем\\yearssalary.xlsx'
data = pd.read_excel(file_path)

X = data[['YearsExperience']].values
y = data['Salary'].values

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение параметров для поиска по сетке
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250]  # Здесь вы можете добавить или изменить значения
}

# Создание базовой модели случайного леса
rf = RandomForestRegressor()

# Инициализация GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)

# Поиск наилучших гиперпараметров
grid_search.fit(X_train, y_train)

# Вывод наилучших гиперпараметров
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

best_model = grid_search.best_estimator_

# Обучение модели с лучшими параметрами с использованием timeit
start_time = timeit.default_timer()

best_model.fit(X_train, y_train)

end_time = timeit.default_timer()
training_time = end_time - start_time

# Предсказание результата
predictions = best_model.predict(X_test)

# Оценка производительности модели с наилучшими гиперпараметрами
accuracy = r2_score(y_test, predictions)
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("Model accuracy:", accuracy)
print("Среднее время обучения:", training_time, "секунд")