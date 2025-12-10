import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, deque

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ==========================
# CONFIGURACIONES
# ==========================

RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

JUGADA_A_NUM = {"r": 0, "p": 1, "s": 2}
NUM_A_JUGADA = {0: "r", 1: "p", 2: "s"}

GANA_A = {"r": "s", "p": "r", "s": "p"}
PIERDE_CONTRA = {"r": "p", "p": "s", "s": "r"}


# =============================================================================
# PARTE 1 – CARGA DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    df = pd.read_csv(ruta_csv)

    columnas_requeridas = {
        "nºPartidas", "Cosmin", "Jugadores", "Probabilidad de Piedra", "Probabilidad de Papel",
        "Probabilidad de Tijera", "Resultado de Partida (cosmin)", "Último movimiento",
        "Patrones dentro de cada 4 jugadas", "Comportamiento tras partida ganada",
        "Comportamiento tras partida perdida", "Comportamiento tras partida empatada", "Entropia"
    }

    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    return df


# =============================================================================
# PARTE 2 – PREPARACIÓN
# =============================================================================

def limpiar_comportamiento(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x in ["r", "p", "s"]:
            return {"r": 0, "p": 1, "s": 2}[x]
        return 0
    return 0


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Probabilidad de Piedra", "Probabilidad de Papel", "Probabilidad de Tijera"]:
        df[col] = df[col].astype(str).str.replace("%", "", regex=False)
        df[col] = df[col].str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df["Cosmin"] = df["Cosmin"].str.strip().str.lower()
    df["Cosmin_num"] = df["Cosmin"].map(JUGADA_A_NUM)

    df = df[df["Cosmin_num"].notna()]

    df["Jugadores"] = df["Jugadores"].str.strip().str.lower()
    df["Jugadores_num"] = df["Jugadores"].map(JUGADA_A_NUM)

    df["Ultimo_num"] = df["Último movimiento"].apply(limpiar_comportamiento)
    df["Gana_num"] = df["Comportamiento tras partida ganada"].apply(limpiar_comportamiento)
    df["Pierde_num"] = df["Comportamiento tras partida perdida"].apply(limpiar_comportamiento)
    df["Empata_num"] = df["Comportamiento tras partida empatada"].apply(limpiar_comportamiento)

    # CRÍTICO: Predecir la PRÓXIMA jugada de Cosmin
    df["proxima_jugada_cosmin"] = df["Cosmin_num"].shift(-1)

    df = df[df["proxima_jugada_cosmin"].notna()]
    df["proxima_jugada_cosmin"] = df["proxima_jugada_cosmin"].astype(int)

    return df.reset_index(drop=True)


# =============================================================================
# PARTE 3 – FEATURE ENGINEERING
# =============================================================================

def calcular_entropia(jugadas):
    if len(jugadas) == 0:
        return 0
    counter = Counter(jugadas)
    total = len(jugadas)
    entropia = 0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropia -= p * np.log2(p)
    return entropia


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Frecuencias acumuladas
    df["freq_r"] = (df["Cosmin_num"] == 0).expanding().mean()
    df["freq_p"] = (df["Cosmin_num"] == 1).expanding().mean()
    df["freq_s"] = (df["Cosmin_num"] == 2).expanding().mean()

    # Lags (últimas jugadas)
    for i in range(1, 6):
        df[f"lag{i}"] = df["Cosmin_num"].shift(i)

    # Entropía móvil
    df["entropia"] = df["Cosmin_num"].rolling(window=10, min_periods=3).apply(
        lambda x: calcular_entropia(x.tolist())
    )

    df = df.fillna(0)
    df = df.iloc[10:]

    return df.reset_index(drop=True)


def seleccionar_features(df: pd.DataFrame):
    features = [
        "Probabilidad de Piedra", "Probabilidad de Papel", "Probabilidad de Tijera",
        "freq_r", "freq_p", "freq_s",
        "lag1", "lag2", "lag3", "lag4", "lag5",
        "Ultimo_num", "Gana_num", "Pierde_num", "Empata_num",
        "Entropia", "entropia"
    ]

    X = df[features]
    y = df["proxima_jugada_cosmin"].astype(int)

    return X, y


# =============================================================================
# PARTE 4 – ENTRENAMIENTO
# =============================================================================

def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    print("\nEntrenando modelo...")
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))

    return rf


def guardar_modelo(modelo, ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)

    print("\nModelo guardado correctamente.")


def cargar_modelo(ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 5 – IA IMPREDECIBLE Y ADAPTATIVA
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo=None):
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            self.columnas = self.modelo.feature_names_in_
        except:
            print("No se encontró modelo. Jugando semi-aleatorio.")
            self.modelo = None
            self.columnas = None

        self.historial_oponente = deque(maxlen=50)
        self.historial_ia = deque(maxlen=20)  # Para evitar auto-patrones
        self.probabilidades_base = [0.33, 0.33, 0.33]
        self.historial_ultimos = [0, 0, 0, 0]
        self.ultima_estrategia = None
        self.contador_estrategia = 0

    def registrar(self, jugada_oponente):
        """Registra la jugada del oponente"""
        self.historial_oponente.append(JUGADA_A_NUM[jugada_oponente])

    def registrar_jugada_ia(self, jugada_ia):
        """Registra la propia jugada de la IA para evitar patrones"""
        self.historial_ia.append(JUGADA_A_NUM[jugada_ia])

    def obtener_features(self):
        historial = np.array(list(self.historial_oponente))
        n = len(historial)

        if n == 0:
            return np.zeros(len(self.columnas))

        freq_r = np.mean(historial == 0)
        freq_p = np.mean(historial == 1)
        freq_s = np.mean(historial == 2)

        lags = []
        for i in range(1, 6):
            if i <= n:
                lags.append(historial[-i])
            else:
                lags.append(np.argmax([freq_r, freq_p, freq_s]))

        ultimos = self.historial_ultimos

        entropia_total = calcular_entropia(historial.tolist())
        entropia_reciente = calcular_entropia(historial[-10:].tolist()) if n >= 10 else entropia_total

        features = [
            self.probabilidades_base[0], self.probabilidades_base[1], self.probabilidades_base[2],
            freq_r, freq_p, freq_s,
            *lags,
            *ultimos,
            entropia_total, entropia_reciente
        ]

        return np.array(features)

    def predecir_proxima_jugada(self):
        """Predice la PRÓXIMA jugada del oponente usando ML"""
        if self.modelo is None:
            return np.random.choice(["r", "p", "s"])

        X = pd.DataFrame([self.obtener_features()], columns=self.columnas)

        try:
            if hasattr(self.modelo, 'predict_proba'):
                probs = self.modelo.predict_proba(X)[0]

                if len(probs) == 2:
                    probs_full = np.zeros(3)
                    classes = self.modelo.classes_
                    for i, clase in enumerate(classes):
                        probs_full[clase] = probs[i]
                    probs = probs_full

                # Añadir algo de ruido para evitar ser predecible
                ruido = np.random.dirichlet([0.3, 0.3, 0.3]) * 0.15
                probs = probs * 0.85 + ruido
                probs = probs / probs.sum()

                pred = np.random.choice([0, 1, 2], p=probs)
            else:
                pred = self.modelo.predict(X)[0]
        except Exception as e:
            pred = np.random.choice([0, 1, 2])

        return NUM_A_JUGADA[pred]

    def hay_patron_ia(self, jugada_propuesta):
        """Detecta si la IA está cayendo en un patrón"""
        if len(self.historial_ia) < 4:
            return False

        ultimas_4_ia = list(self.historial_ia)[-4:]
        jugada_num = JUGADA_A_NUM[jugada_propuesta]

        # Detectar patrón de "doble-doble" (r,r,p,p,s,s)
        if len(ultimas_4_ia) >= 4:
            if (ultimas_4_ia[-1] == ultimas_4_ia[-2] and
                    ultimas_4_ia[-3] == ultimas_4_ia[-4] and
                    ultimas_4_ia[-1] != ultimas_4_ia[-3]):
                # Estamos en un patrón doble-doble
                return True

        # Detectar si estamos repitiendo demasiado la misma jugada
        ultimas_3_ia = list(self.historial_ia)[-3:]
        if len(ultimas_3_ia) >= 3:
            counter = Counter(ultimas_3_ia)
            if counter.most_common(1)[0][1] >= 3:
                return True

        return False

    def decidir_jugada(self):
        """
        ESTRATEGIA ADAPTATIVA E IMPREDECIBLE:

        1. Detecta patrones del oponente con confianza variable
        2. Usa ML con probabilidades (no determinista)
        3. Evita caer en patrones propios
        4. Mezcla estrategias de forma aleatoria
        """
        n = len(self.historial_oponente)

        # Lista de posibles estrategias con pesos
        estrategias = []

        # === ESTRATEGIA 1: Repetición inmediata (FUERTE) ===
        if n >= 2 and self.historial_oponente[-1] == self.historial_oponente[-2]:
            jugada_repetida = NUM_A_JUGADA[self.historial_oponente[-1]]
            estrategias.append(("repeticion", GANA_A[jugada_repetida], 0.6))

        # === ESTRATEGIA 2: Frecuencia en últimas 5 (MODERADA) ===
        if n >= 5:
            ultimas_5 = list(self.historial_oponente)[-5:]
            counter = Counter(ultimas_5)
            jugada_mas_comun, count = counter.most_common(1)[0]

            if count >= 3:
                estrategias.append(("frecuencia_5", GANA_A[NUM_A_JUGADA[jugada_mas_comun]], 0.5))

        # === ESTRATEGIA 3: Tendencia en últimas 3 (SUAVE) ===
        if n >= 3:
            ultimas_3 = list(self.historial_oponente)[-3:]
            counter_3 = Counter(ultimas_3)

            for jugada, count in counter_3.items():
                if count >= 2:
                    estrategias.append(("tendencia_3", GANA_A[NUM_A_JUGADA[jugada]], 0.35))
                    break

        # === ESTRATEGIA 4: Predicción ML (SIEMPRE DISPONIBLE) ===
        pred_ml = self.predecir_proxima_jugada()
        estrategias.append(("ml", GANA_A[pred_ml], 0.4))

        # === ESTRATEGIA 5: Semi-aleatorio (ANTI-PATRON) ===
        if n >= 10:
            historial = np.array(list(self.historial_oponente))
            freq_r = np.mean(historial == 0)
            freq_p = np.mean(historial == 1)
            freq_s = np.mean(historial == 2)

            # Invertir frecuencias: jugar contra lo que MENOS juega
            freq_inversa = [1 - freq_r, 1 - freq_p, 1 - freq_s]
            freq_inversa = np.array(freq_inversa) / sum(freq_inversa)

            jugada_inversa = np.random.choice(["r", "p", "s"], p=freq_inversa)
            estrategias.append(("anti_patron", jugada_inversa, 0.25))

        # Seleccionar estrategia usando pesos probabilísticos
        if estrategias:
            nombres, jugadas, pesos = zip(*estrategias)
            pesos = np.array(pesos)
            pesos = pesos / pesos.sum()

            idx = np.random.choice(len(estrategias), p=pesos)
            estrategia_elegida, jugada_elegida = nombres[idx], jugadas[idx]

            # Anti-patrón: verificar si la IA está siendo predecible
            if self.hay_patron_ia(jugada_elegida):
                # Forzar variación
                todas_jugadas = ["r", "p", "s"]
                if len(self.historial_ia) >= 2:
                    # Evitar las últimas 2 jugadas
                    evitar = [NUM_A_JUGADA[self.historial_ia[-1]],
                              NUM_A_JUGADA[self.historial_ia[-2]]]
                    opciones = [j for j in todas_jugadas if j not in evitar]
                    if opciones:
                        jugada_elegida = np.random.choice(opciones)

            # Registrar para tracking
            self.ultima_estrategia = estrategia_elegida
            self.registrar_jugada_ia(jugada_elegida)

            return jugada_elegida

        # Fallback: aleatorio puro
        jugada = np.random.choice(["r", "p", "s"])
        self.registrar_jugada_ia(jugada)
        return jugada


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Cargando datos…")
    df = cargar_datos()

    print("Preparando datos...")
    df = preparar_datos(df)

    print("Creando features...")
    df = crear_features(df)

    X, y = seleccionar_features(df)

    print(f"\nDataset final: {X.shape[0]} muestras, {X.shape[1]} features")
    print("\nEntrenando modelo ADAPTATIVO E IMPREDECIBLE…")
    modelo = entrenar_modelo(X, y)

    guardar_modelo(modelo)
    print("\n✓ Entrenamiento finalizado!")
    print("\n ESTRATEGIA ADAPTATIVA:")
    print("  Mezcla estrategias con pesos probabilísticos")
    print("  Detecta patrones del oponente")
    print("  Evita caer en patrones propios")
    print("  Usa ML con ruido aleatorio")
    print("  Cambia dinámicamente de estrategia")


if __name__ == "__main__":
    main()