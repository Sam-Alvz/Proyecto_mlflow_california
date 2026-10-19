# Proyecto MLflow: Predicción de Precios de Viviendas en California

Este repositorio contiene la implementación de un pipeline de Machine Learning para estimar los precios medios de las viviendas en California (*California Housing Dataset*), integrando **MLflow** para el rastreo de experimentos, registro de métricas e hiperparámetros, y la gestión del ciclo de vida del modelo (MLOps).

---

## 🛠️ Estructura del Proyecto

```text
proyecto_mlflow/
│
├── entrena.py            # Script principal de entrenamiento y registro en MLflow
├── mlflow.db             # Base de datos SQLite local de experimentos
├── mlruns/               # Artefactos y registros detallados de MLflow
├── datos/                # Conjunto de datos e insumos
├── fuentes/              # Código fuente auxiliar y módulos
├── .gitignore            # Archivo de exclusiones de Git
└── README.md             # Documentación principal del repositorio
```

---

## 📊 Modelos Comparados y Resultados

Se evaluaron dos algoritmos sobre el conjunto de prueba para seleccionar la mejor arquitectura:

| Modelo | MAE | RMSE | $R^2$ | Configuración / Hiperparámetros |
| :--- | :---: | :---: | :---: | :--- |
| **Regresión Lineal (Baseline)** | 0.5332 | 0.7456 | 0.5758 | Configuración estándar (`sklearn`) |
| **XGBoost Regressor** | **0.3297** | **0.4936** | **0.8141** | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5` |

**Conclusión:** XGBoost redujo el error absoluto medio ($MAE$) en un **38%** respecto a la Regresión Lineal, explicando más del **81.4%** de la variancia de los datos.

---

## 🚀 Instrucciones de Ejecución Local

### 1. Requisitos Previos
Asegúrate de contar con Python 3.11 instalado y ejecutar la instalación de las librerías necesarias:

```bash
pip install mlflow pandas scikit-learn xgboost
```

### 2. Ejecutar el Entrenamiento
Para procesar los datos, entrenar ambos modelos y registrar la ejecución en MLflow:

```bash
python entrena.py
```

### 3. Visualizar en la Interfaz de MLflow (MLflow UI)
Para iniciar el servidor local y revisar las métricas tabuladas:

```bash
mlflow ui
```

Accede desde cualquier navegador web a la dirección: **`http://localhost:5000`**