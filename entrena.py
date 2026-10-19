import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Configurar el experimento en MLflow
mlflow.set_experiment("Prediccion_California_Housing")

# 2. Cargar datos de California Housing
print("Cargando conjunto de datos...")
housing = fetch_california_housing(as_frame=True)
X = housing.frame.drop(columns=['MedHouseVal'])
y = housing.frame['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODELO 1: Regresión Lineal ---
print("Entrenando Regresión Lineal...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

with mlflow.start_run(run_name="Modelo_Regresion_Lineal"):
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred_lr))
    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    mlflow.log_metric("R2", r2_score(y_test, y_pred_lr))

# --- MODELO 2: XGBoost Regressor ---
print("Entrenando XGBoost...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

with mlflow.start_run(run_name="Modelo_XGBoost"):
    mlflow.log_param("model_type", "XGBRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred_xgb))
    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
    mlflow.log_metric("R2", r2_score(y_test, y_pred_xgb))

print("\n¡Experimentos registrados exitosamente en MLflow!")