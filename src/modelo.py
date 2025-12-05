import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, deque

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

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

    df["proxima_jugada_j2"] = df["Cosmin_num"].shift(-1)

    df = df[df["proxima_jugada_j2"].notna()]
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)

    return df.reset_index(drop=True)


# =============================================================================
# PARTE 3 – FEATURE ENGINEERING MEJORADO
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


def detectar_racha_actual(jugadas, ventana=5):
    if len(jugadas) < ventana:
        return 0, 0

    ultimas = jugadas[-ventana:]
    if len(set(ultimas)) == 1:
        jugada_racha = ultimas[0]
        longitud = ventana
        idx = len(jugadas) - ventana - 1
        while idx >= 0 and jugadas[idx] == jugada_racha:
            longitud += 1
            idx -= 1
        return longitud, jugada_racha
    return 0, 0


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["freq_r"] = (df["Cosmin_num"] == 0).expanding().mean()
    df["freq_p"] = (df["Cosmin_num"] == 1).expanding().mean()
    df["freq_s"] = (df["Cosmin_num"] == 2).expanding().mean()

    for i in range(1, 8):
        df[f"lag{i}"] = df["Cosmin_num"].shift(i)

    df["freq_r_recent"] = df["Cosmin_num"].rolling(window=10, min_periods=1).apply(
        lambda x: (x == 0).mean()
    )
    df["freq_p_recent"] = df["Cosmin_num"].rolling(window=10, min_periods=1).apply(
        lambda x: (x == 1).mean()
    )
    df["freq_s_recent"] = df["Cosmin_num"].rolling(window=10, min_periods=1).apply(
        lambda x: (x == 2).mean()
    )

    rachas_info = []
    for idx in range(len(df)):
        if idx < 5:
            rachas_info.append((0, 0))
        else:
            jugadas_previas = df["Cosmin_num"].iloc[:idx + 1].tolist()
            longitud, jugada = detectar_racha_actual(jugadas_previas, ventana=3)
            rachas_info.append((longitud, jugada))

    df["racha_longitud"] = [x[0] for x in rachas_info]
    df["racha_jugada"] = [x[1] for x in rachas_info]

    df["entropia_reciente"] = df["Cosmin_num"].rolling(window=10, min_periods=3).apply(
        lambda x: calcular_entropia(x.tolist())
    )

    df["tendencia_r"] = df["freq_r_recent"] - df["freq_r_recent"].shift(5)
    df["tendencia_p"] = df["freq_p_recent"] - df["freq_p_recent"].shift(5)
    df["tendencia_s"] = df["freq_s_recent"] - df["freq_s_recent"].shift(5)

    df["alterna_rp"] = ((df["lag1"] == 0) & (df["lag2"] == 1) |
                        (df["lag1"] == 1) & (df["lag2"] == 0)).astype(int)
    df["alterna_rs"] = ((df["lag1"] == 0) & (df["lag2"] == 2) |
                        (df["lag1"] == 2) & (df["lag2"] == 0)).astype(int)
    df["alterna_ps"] = ((df["lag1"] == 1) & (df["lag2"] == 2) |
                        (df["lag1"] == 2) & (df["lag2"] == 1)).astype(int)

    df["diversidad_5"] = df["Cosmin_num"].rolling(window=5, min_periods=1).apply(
        lambda x: len(set(x))
    )

    df["desv_equilibrio_r"] = (df["freq_r_recent"] - 0.333).abs()
    df["desv_equilibrio_p"] = (df["freq_p_recent"] - 0.333).abs()
    df["desv_equilibrio_s"] = (df["freq_s_recent"] - 0.333).abs()

    df["diff_r"] = df["freq_r_recent"] - df["freq_r"]
    df["diff_p"] = df["freq_p_recent"] - df["freq_p"]
    df["diff_s"] = df["freq_s_recent"] - df["freq_s"]

    df = df.fillna(0)
    df = df.iloc[10:]

    return df.reset_index(drop=True)


# =============================================================================
# PARTE 4 – SELECCIÓN DE FEATURES MEJORADA
# =============================================================================

def seleccionar_features(df: pd.DataFrame):
    features = [
        "Probabilidad de Piedra", "Probabilidad de Papel", "Probabilidad de Tijera",
        "freq_r", "freq_p", "freq_s",
        "freq_r_recent", "freq_p_recent", "freq_s_recent",
        "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7",
        "Ultimo_num", "Gana_num", "Pierde_num", "Empata_num",
        "racha_longitud", "racha_jugada",
        "Entropia", "entropia_reciente",
        "tendencia_r", "tendencia_p", "tendencia_s",
        "alterna_rp", "alterna_rs", "alterna_ps",
        "diversidad_5",
        "desv_equilibrio_r", "desv_equilibrio_p", "desv_equilibrio_s",
        "diff_r", "diff_p", "diff_s"
    ]

    X = df[features]
    y = df["proxima_jugada_j2"].astype(int)

    return X, y


# =============================================================================
# PARTE 5 – ENTRENAMIENTO CON ENSEMBLE MEJORADO
# =============================================================================

def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )

    rf1 = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42
    )

    rf2 = RandomForestClassifier(
        n_estimators=500,
        max_depth=35,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=123
    )

    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.08,
        random_state=42
    )

    modelo = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('gb', gb)],
        voting='soft'
    )

    print("\nEntrenando modelo ensemble...")
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))

    return modelo


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
# PARTE 6 – IA ULTRA-ADAPTATIVA CON MÚLTIPLES DETECTORES
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo=None):
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            self.columnas = self.modelo.feature_names_in_
        except:
            print("No se encontró modelo. Jugando aleatorio.")
            self.modelo = None
            self.columnas = None

        self.historial_oponente = deque(maxlen=50)
        self.historial_ia = deque(maxlen=50)
        self.historial_resultados = deque(maxlen=50)
        self.historial_ultimos = [0, 0, 0, 0]
        self.probabilidades_base = [0.33, 0.33, 0.33]

        self.racha_detectada = None
        self.jugadas_desde_cambio = 0

        # Variables para patrones de empate
        self.empates_consecutivos = 0
        self.ultima_jugada_empate = None

        # NUEVO: Historial de patrones detectados y éxito
        self.patron_exitoso = None  # Guarda qué patrón funcionó
        self.confianza_patron = 0.5  # Nivel de confianza en patrones

        # NUEVO: Detector de comportamiento tras victoria/derrota
        self.ultima_victoria_ia = False
        self.ultima_derrota_ia = False

    def registrar(self, jugada_oponente, jugada_ia=None):
        """Registra la jugada del oponente y el resultado"""
        self.historial_oponente.append(JUGADA_A_NUM[jugada_oponente])

        if jugada_ia:
            self.historial_ia.append(JUGADA_A_NUM[jugada_ia])

            # Determinar resultado
            if jugada_oponente == jugada_ia:
                resultado = "empate"
                self.empates_consecutivos += 1
                self.ultima_jugada_empate = jugada_oponente
                self.ultima_victoria_ia = False
                self.ultima_derrota_ia = False
            elif GANA_A[jugada_ia] == jugada_oponente:
                resultado = "win"
                self.ultima_victoria_ia = True
                self.ultima_derrota_ia = False
                if self.empates_consecutivos > 0:
                    self.empates_consecutivos = 0
                    self.ultima_jugada_empate = None
            else:
                resultado = "loss"
                self.ultima_victoria_ia = False
                self.ultima_derrota_ia = True
                if self.empates_consecutivos > 0:
                    self.empates_consecutivos = 0
                    self.ultima_jugada_empate = None

            self.historial_resultados.append(resultado)

        self._actualizar_racha(jugada_oponente)

    def _actualizar_racha(self, jugada_oponente):
        """Detecta si el oponente está en racha"""
        if len(self.historial_oponente) < 2:
            return

        if self.historial_oponente[-1] == self.historial_oponente[-2]:
            if self.racha_detectada and self.racha_detectada[0] == jugada_oponente:
                self.racha_detectada = (jugada_oponente, self.racha_detectada[1] + 1)
            else:
                self.racha_detectada = (jugada_oponente, 2)
            self.jugadas_desde_cambio = 0
        else:
            self.racha_detectada = None
            self.jugadas_desde_cambio += 1

    def _detectar_patron_empates(self):
        """Detecta patrón tras empates consecutivos"""
        if self.empates_consecutivos >= 2 and self.ultima_jugada_empate:
            return PIERDE_CONTRA[self.ultima_jugada_empate]

        if self.empates_consecutivos == 1 and self.ultima_jugada_empate:
            if np.random.random() < 0.4:
                return self.ultima_jugada_empate

        return None

    def _detectar_patron_tras_resultado(self):
        """NUEVO: Detecta cambios de comportamiento tras ganar/perder"""
        if len(self.historial_oponente) < 3:
            return None

        # Patrón: tras perder, el oponente tiende a cambiar a lo que HABRÍA ganado
        if self.ultima_victoria_ia and len(self.historial_oponente) >= 2:
            jugada_perdedora = NUM_A_JUGADA[self.historial_oponente[-1]]
            # Predice que jugará lo que le habría ganado a la IA
            if len(self.historial_ia) >= 1:
                jugada_ia_anterior = NUM_A_JUGADA[self.historial_ia[-1]]
                # El oponente puede jugar lo que gana a lo que jugó la IA
                return PIERDE_CONTRA[jugada_ia_anterior]

        # Patrón: tras ganar, a veces repite
        if self.ultima_derrota_ia and len(self.historial_oponente) >= 1:
            if np.random.random() < 0.35:
                return NUM_A_JUGADA[self.historial_oponente[-1]]

        return None

    def _detectar_ciclo_rps(self):
        """NUEVO: Detecta si el oponente está jugando el ciclo r→p→s"""
        if len(self.historial_oponente) < 3:
            return None

        ultimas_3 = [self.historial_oponente[-3], self.historial_oponente[-2], self.historial_oponente[-1]]

        # Ciclo ascendente: r(0) → p(1) → s(2)
        if ultimas_3 == [0, 1, 2]:
            return "r"  # Predice que volverá a r

        # Ciclo parcial detectado
        if len(self.historial_oponente) >= 2:
            if self.historial_oponente[-2] == 0 and self.historial_oponente[-1] == 1:
                if np.random.random() < 0.6:
                    return "s"  # Probablemente jugará s
            elif self.historial_oponente[-2] == 1 and self.historial_oponente[-1] == 2:
                if np.random.random() < 0.6:
                    return "r"  # Probablemente jugará r

        return None

    def _detectar_patron_simple(self):
        """Detecta patrones simples de alternancia"""
        if len(self.historial_oponente) < 4:
            return None

        # Patrón AB-AB
        if (self.historial_oponente[-1] == self.historial_oponente[-3] and
                self.historial_oponente[-2] == self.historial_oponente[-4] and
                self.historial_oponente[-1] != self.historial_oponente[-2]):
            return NUM_A_JUGADA[self.historial_oponente[-3]]

        # Patrón ABC-ABC
        if len(self.historial_oponente) >= 6:
            if (self.historial_oponente[-1] == self.historial_oponente[-4] and
                    self.historial_oponente[-2] == self.historial_oponente[-5] and
                    self.historial_oponente[-3] == self.historial_oponente[-6]):
                return NUM_A_JUGADA[self.historial_oponente[-3]]

        return None

    def _detectar_sesgo_frecuencia(self):
        """NUEVO: Detecta si el oponente tiene sesgo hacia alguna jugada"""
        if len(self.historial_oponente) < 10:
            return None

        recientes = list(self.historial_oponente)[-10:]
        counter = Counter(recientes)

        # Si una jugada aparece 6+ veces en las últimas 10
        for jugada, count in counter.items():
            if count >= 6:
                # Predice que seguirá con esa tendencia
                if np.random.random() < 0.65:
                    return NUM_A_JUGADA[jugada]

        return None

    def obtener_features(self):
        historial = np.array(list(self.historial_oponente))
        n = len(historial)

        if n == 0:
            return np.zeros(len(self.columnas))

        freq_r = np.mean(historial == 0)
        freq_p = np.mean(historial == 1)
        freq_s = np.mean(historial == 2)

        recientes = historial[-10:] if n >= 10 else historial
        freq_r_recent = np.mean(recientes == 0)
        freq_p_recent = np.mean(recientes == 1)
        freq_s_recent = np.mean(recientes == 2)

        lags = []
        for i in range(1, 8):
            if i <= n:
                lags.append(historial[-i])
            else:
                lags.append(np.argmax([freq_r, freq_p, freq_s]))

        ultimos = self.historial_ultimos

        longitud_racha, jugada_racha = detectar_racha_actual(historial.tolist(), ventana=3)

        entropia_total = calcular_entropia(historial.tolist())
        entropia_reciente = calcular_entropia(recientes.tolist())

        if n >= 10:
            ultimas_5 = historial[-5:]
            previas_5 = historial[-10:-5]
            tend_r = np.mean(ultimas_5 == 0) - np.mean(previas_5 == 0)
            tend_p = np.mean(ultimas_5 == 1) - np.mean(previas_5 == 1)
            tend_s = np.mean(ultimas_5 == 2) - np.mean(previas_5 == 2)
        else:
            tend_r = tend_p = tend_s = 0

        alterna_rp = int(n >= 2 and (
                (historial[-1] == 0 and historial[-2] == 1) or
                (historial[-1] == 1 and historial[-2] == 0)
        ))
        alterna_rs = int(n >= 2 and (
                (historial[-1] == 0 and historial[-2] == 2) or
                (historial[-1] == 2 and historial[-2] == 0)
        ))
        alterna_ps = int(n >= 2 and (
                (historial[-1] == 1 and historial[-2] == 2) or
                (historial[-1] == 2 and historial[-2] == 1)
        ))

        diversidad = len(set(recientes))

        desv_r = abs(freq_r_recent - 0.333)
        desv_p = abs(freq_p_recent - 0.333)
        desv_s = abs(freq_s_recent - 0.333)

        diff_r = freq_r_recent - freq_r
        diff_p = freq_p_recent - freq_p
        diff_s = freq_s_recent - freq_s

        features = [
            self.probabilidades_base[0], self.probabilidades_base[1], self.probabilidades_base[2],
            freq_r, freq_p, freq_s,
            freq_r_recent, freq_p_recent, freq_s_recent,
            *lags,
            *ultimos,
            longitud_racha, jugada_racha,
            entropia_total, entropia_reciente,
            tend_r, tend_p, tend_s,
            alterna_rp, alterna_rs, alterna_ps,
            diversidad,
            desv_r, desv_p, desv_s,
            diff_r, diff_p, diff_s
        ]

        return np.array(features)

    def predecir(self):
        if self.modelo is None:
            return np.random.choice(["r", "p", "s"])

        import pandas as pd
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

                noise = np.random.random(3) * 0.02
                probs = probs + noise
                probs = probs / probs.sum()
                pred = np.argmax(probs)
            else:
                pred = self.modelo.predict(X)[0]
        except Exception as e:
            pred = np.random.choice([0, 1, 2])

        return NUM_A_JUGADA[pred]

    def decidir_jugada(self):
        """
        Estrategia ULTRA-ADAPTATIVA con múltiples detectores:
        0. Detectar sesgo de frecuencia (65%)
        1. Detectar patrón tras empates (90%)
        2. Detectar comportamiento tras victoria/derrota (80%)
        3. Detectar ciclos r→p→s (60%)
        4. Detectar rachas largas (90%+)
        5. Detectar patrones de alternancia (65%)
        6. Predicción del modelo (70%)
        7. Variación aleatoria (30%)
        """

        # ===== NIVEL 0: SESGO DE FRECUENCIA =====
        sesgo = self._detectar_sesgo_frecuencia()
        if sesgo and np.random.random() < 0.65:
            return PIERDE_CONTRA[sesgo]

        # ===== NIVEL 1: PATRÓN DE EMPATES =====
        patron_empate = self._detectar_patron_empates()
        if patron_empate and np.random.random() < 0.90:
            return PIERDE_CONTRA[patron_empate]

        # ===== NIVEL 2: COMPORTAMIENTO TRAS RESULTADO =====
        patron_resultado = self._detectar_patron_tras_resultado()
        if patron_resultado and np.random.random() < 0.80:
            return PIERDE_CONTRA[patron_resultado]

        # ===== NIVEL 3: CICLO RPS =====
        ciclo = self._detectar_ciclo_rps()
        if ciclo and np.random.random() < 0.60:
            return PIERDE_CONTRA[ciclo]

        # ===== NIVEL 4: CONTRAATAQUE A RACHAS =====
        if self.racha_detectada and self.racha_detectada[1] >= 2:
            jugada_racha = self.racha_detectada[0]
            longitud = self.racha_detectada[1]

            if longitud >= 5:
                probabilidad = 0.95
            elif longitud >= 3:
                probabilidad = 0.88
            else:
                probabilidad = 0.80

            if np.random.random() < probabilidad:
                return PIERDE_CONTRA[jugada_racha]

        # ===== NIVEL 5: PATRONES DE ALTERNANCIA =====
        patron = self._detectar_patron_simple()
        if patron and np.random.random() < 0.65:
            return PIERDE_CONTRA[patron]

        # ===== NIVEL 6: PREDICCIÓN DEL MODELO =====
        pred = self.predecir()

        if np.random.random() < 0.70:
            return PIERDE_CONTRA[pred]
        else:
            # Variación estratégica
            otras = [x for x in ["r", "p", "s"] if x != pred]
            pred_alternativa = np.random.choice(otras)
            return PIERDE_CONTRA[pred_alternativa]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Cargando datos…")
    df = cargar_datos()

    print("Preparando datos...")
    df = preparar_datos(df)

    print("Creando features mejoradas...")
    df = crear_features(df)

    X, y = seleccionar_features(df)

    print(f"\nDataset final: {X.shape[0]} muestras, {X.shape[1]} features")
    print("\nEntrenando modelo ULTRA mejorado…")
    modelo = entrenar_modelo(X, y)

    guardar_modelo(modelo)
    print("\n✓ Entrenamiento finalizado con éxito!")
    print("\n🎯 ESTRATEGIA ULTRA-ADAPTATIVA ACTIVADA:")
    print("  0. DETECTOR DE SESGO: Frecuencia alta en últimas 10 jugadas (65%)")
    print("  1. DETECTOR DE EMPATES: Tras 2+ empates consecutivos (90%)")
    print("  2. DETECTOR POST-RESULTADO: Comportamiento tras ganar/perder (80%)")
    print("  3. DETECTOR DE CICLOS: Secuencia r→p→s (60%)")
    print("  4. CONTRAATAQUE A RACHAS: 2+ repeticiones (80-95%)")
    print("  5. PATRONES DE ALTERNANCIA: AB-AB, ABC-ABC (65%)")
    print("  6. PREDICCIÓN ML: Modelo ensemble mejorado (70%)")
    print("  7. VARIACIÓN ESTRATÉGICA: Para impredecibilidad (30%)")

if __name__ == "__main__":
    main()