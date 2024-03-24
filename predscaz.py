from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from analyzis import house
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np


# Предполагается, что датафрейм 'house' уже определен
X = house.drop('PRICE', axis=1)
y = house['PRICE']

# Установить seed для репликабельности результатов
np.random.seed(42)

# Разделить данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['ADDRESS', 'STREET_NAME', 'LONG_NAME']

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])

pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LinearRegression())])

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestRegressor())])

# Установить seed для репликабельности результатов
np.random.seed(42)

# Обучить модели
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# Получить предсказания
predictions_lr = pipeline_lr.predict(X_test)
predictions_rf = pipeline_rf.predict(X_test)

# Оценить модели
scores_lr = cross_val_score(pipeline_lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
scores_rf = cross_val_score(pipeline_rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

print("Средняя точность линейной регрессии: ", scores_lr.mean())
print("Средняя точность случайного леса: ", scores_rf.mean())


mse_lr = mean_squared_error(y_test, predictions_lr)
mse_rf = mean_squared_error(y_test, predictions_rf)

print("Среднеквадратичная ошибка линейной регрессии: ", mse_lr)
print("Среднеквадратичная ошибка случайного леса: ", mse_rf)