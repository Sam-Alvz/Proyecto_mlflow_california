import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def cargar_datos_iniciales():
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    df.rename(columns={'MedHouseVal': 'PrecioMedio'}, inplace=True)
    return df

def limpiar_y_preparar_datos(df):
    df_limpio = df.copy()
    df_limpio = df_limpio.drop_duplicates()
    df_limpio = df_limpio.replace([np.inf, -np.inf], np.nan)
    df_limpio = df_limpio.dropna()

    X = df_limpio.drop(columns=['PrecioMedio'])
    y = df_limpio['PrecioMedio']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return df_limpio, X_scaled, y, scaler