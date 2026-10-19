### 🚀 Añadido
- *Pipeline de Entrenamiento y Evaluación:* Implementación del script entrena.py que integra la carga del dataset California Housing, división de conjuntos (train/test) y entrenamiento de modelos.
- *Rastreo de Experimentos con MLflow:*
  - Configuración del experimento Prediccion_California_Housing.
  - Registro automático de métricas de desempeño ($MAE$, $RMSE$, $R^2$).
  - Registro de hiperparámetros y metadatos de los modelos evaluados.
  - Almacenamiento local de artefactos y base de datos mlflow.db.
- *Evaluación Comparativa de Modelos:*
  - Modelo_Regresion_Lineal: Modelo base (baseline) con $R^2 = 0.5758$ y $MAE = 0.5332$.
  - Modelo_XGBoost: Modelo de ensamble optimizado con $R^2 = 0.8141$ y $MAE = 0.3297$.
- *Documentación del Repositorio:*
  - Creación de README.md detallado con requisitos, estructura y comandos de ejecución.
  - Inclusión de .gitignore para exclusión de entornos virtuales y archivos temporales.

---

## [0.1.0] - 2026-10-17

### 🛠️ Inicial
- Estructuración inicial de carpetas del proyecto (datos/, fuentes/).
- Preparación del entorno de desarrollo local en Python 3.11.
