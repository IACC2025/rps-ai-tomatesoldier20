import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Relación de qué gana a qué
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1 – EXTRACCIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"El archivo '{ruta_csv}' no existe.")

    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:
        raise RuntimeError(f"Error al leer el CSV: {e}")

    columnas_necesarias = {"jugada_j1", "jugada_j2"}
    faltantes = columnas_necesarias - set(df.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertir jugadas a números
    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    # Target: qué jugada hará j2 en la siguiente ronda
    df["proxima_jugada_j2"] = df["jugada_j2_num"].shift(-1)

    # Eliminar filas con valores nulos
    df = df.dropna().reset_index(drop=True)

    return df


# =============================================================================
# PARTE 2 – FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ================================
    # Feature 1: Frecuencia histórica
    # ================================
    df["freq_piedra"] = (df["jugada_j2_num"] == 0).expanding().mean()
    df["freq_papel"] = (df["jugada_j2_num"] == 1).expanding().mean()
    df["freq_tijera"] = (df["jugada_j2_num"] == 2).expanding().mean()

    # ================================
    # Feature 2: Últimas jugadas (lags)
    # ================================
    df["j2_lag1"] = df["jugada_j2_num"].shift(1)
    df["j2_lag2"] = df["jugada_j2_num"].shift(2)
    df["j2_lag3"] = df["jugada_j2_num"].shift(3)

    # ================================
    # Feature 3: Resultado previo
    # ================================
    def resultado(j1, j2):
        if j1 == j2: return 0
        if GANA_A[NUM_A_JUGADA[j1]] == NUM_A_JUGADA[j2]:
            return 1
        return -1

    df["resultado_prev"] = [
        resultado(df["jugada_j1_num"][i-1], df["jugada_j2_num"][i-1]) if i > 0 else 0
        for i in range(len(df))
    ]

    # Eliminar nulos generados por shift
    df = df.dropna().reset_index(drop=True)

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    feature_cols = [
        "freq_piedra", "freq_papel", "freq_tijera",
        "j2_lag1", "j2_lag2", "j2_lag3",
        "resultado_prev"
    ]

    X = df[feature_cols]
    y = df["proxima_jugada_j2"].astype(int)

    return X, y


# =============================================================================
# PARTE 3 – ENTRENAMIENTO
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    mejor_modelo = None
    mejor_accuracy = -1

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)

        print(f"\n=== Modelo: {nombre} ===")
        print("Accuracy:", acc)
        print(classification_report(y_test, pred))

        if acc > mejor_accuracy:
            mejor_accuracy = acc
            mejor_modelo = modelo

    print(f"\n>>> Mejor modelo: {type(mejor_modelo).__name__} (Accuracy={mejor_accuracy})")
    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)

    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError("Modelo no encontrado")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4 – IA PARA JUGAR
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo: str = None):
        try:
            self.modelo = cargar_modelo(ruta_modelo)
        except:
            print("Modelo no encontrado. Jugará aleatorio.")
            self.modelo = None

        self.historial = []

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        if len(self.historial) == 0:
            return np.zeros(7)

        jugadas_j2 = [JUGADA_A_NUM[j2] for _, j2 in self.historial]

        freq = [
            np.mean([j == 0 for j in jugadas_j2]),
            np.mean([j == 1 for j in jugadas_j2]),
            np.mean([j == 2 for j in jugadas_j2])
        ]

        lags = [
            jugadas_j2[-1] if len(jugadas_j2) >= 1 else 1,
            jugadas_j2[-2] if len(jugadas_j2) >= 2 else 1,
            jugadas_j2[-3] if len(jugadas_j2) >= 3 else 1,
        ]

        return np.array(freq + lags + [0])

    def predecir_jugada_oponente(self) -> str:
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales().reshape(1, -1)
        pred = self.modelo.predict(features)[0]
        return NUM_A_JUGADA[pred]

    def decidir_jugada(self) -> str:
        pred = self.predecir_jugada_oponente()
        return PIERDE_CONTRA[pred]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    df = cargar_datos()
    df = preparar_datos(df)
    df = crear_features(df)
    X, y = seleccionar_features(df)

    modelo = entrenar_modelo(X, y)
    guardar_modelo(modelo)

    print("\nEntrenamiento terminado.")


if __name__ == "__main__":
    main()
