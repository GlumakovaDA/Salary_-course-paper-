import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
import timeit

file_path = 'C:\\Users\\USER\\Desktop\\курсовая_работа2сем\\yearssalary.xlsx'
data = pd.read_excel(file_path)
x = data[['YearsExperience']]
y = data['Salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Определение модели
model = Ridge()

# Параметры для настройки
parameters = {'alpha': [1e-2, 0.1, 1, 5, 10, 20, 50]}

# Создание объекта GridSearchCV
clf = GridSearchCV(model, parameters, cv=10)

clf.fit(x_train, y_train)

# Вывод лучших параметров
print("Лучшие параметры:", clf.best_params_)

# Использование лучших параметров для модели
best_model = clf.best_estimator_

# Обучение модели с лучшими параметрами
start_time = timeit.default_timer()

best_model.fit(x_train, y_train)

end_time = timeit.default_timer()
training_time = end_time - start_time

# Предсказание результатов с полученной моделью
predictions = best_model.predict(x_test)

# Вычисление MAE, MSE и точности модели
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("Model accuracy:", r2_score(y_test, predictions))
print("Среднее время обучения:", training_time)

plt.scatter(x_test, y_test)
plt.plot(x_train, best_model.predict(x_train), color = 'black')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()