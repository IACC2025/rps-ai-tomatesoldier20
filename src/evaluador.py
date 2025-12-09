"""
RPSAI - Evaluador de Winrate CORREGIDO
=======================================

Este script evalua el rendimiento de tu modelo de IA
jugando partidas contra un humano y calculando el winrate.

CORRECCIÓN CRÍTICA:
- La IA decide ANTES de que el humano juegue
- La IA predice la PRÓXIMA jugada del humano
- El orden es correcto: IA predice → Humano juega → Comparar

Uso:
    python src/evaluador_corregido.py
"""

import sys
from pathlib import Path

# Agregar el directorio src al path para importar modelo
sys.path.insert(0, str(Path(__file__).parent))

from modelo import JugadorIA, JUGADA_A_NUM, NUM_A_JUGADA, GANA_A


# Mapeo de entrada a jugada
ENTRADA_A_JUGADA = {
    "1": "r", "2": "p", "3": "s",
    "p": "r", "a": "p", "t": "s",
    "piedra": "r", "papel": "p", "tijera": "s"
}


def obtener_nota(winrate: float) -> tuple:
    """
    Calcula la nota basada en el winrate.

    Returns:
        (nota, mensaje_especial)
    """
    if winrate >= 55:
        return 10, "BONUS: Fallos en examen no restan!"
    elif winrate >= 50:
        return 10, None
    elif winrate >= 49:
        return 9, None
    elif winrate >= 48:
        return 8, None
    elif winrate >= 46:
        return 7, None
    elif winrate >= 44:
        return 6, None
    elif winrate >= 42:
        return 5, None
    elif winrate >= 40:
        return 4, None
    elif winrate >= 39:
        return 3, None
    elif winrate >= 37:
        return 2, None
    elif winrate >= 35:
        return 1, None
    else:
        return 0, None


def obtener_resultado(jugada_ia: str, jugada_humano: str) -> str:
    """Obtiene el resultado desde la perspectiva de la IA."""
    if jugada_ia == jugada_humano:
        return "empate"
    elif GANA_A[jugada_ia] == jugada_humano:
        return "victoria"
    else:
        return "derrota"


def leer_jugada_humano() -> str:
    """Lee la jugada del humano."""
    while True:
        print("\n[1/p] Piedra  [2/a] Papel  [3/t] Tijera")
        entrada = input("Tu jugada: ").lower().strip()

        if entrada in ENTRADA_A_JUGADA:
            return ENTRADA_A_JUGADA[entrada]

        print("Jugada no valida. Intenta de nuevo.")


def mostrar_ronda(ronda: int, jugada_ia: str, jugada_humano: str, resultado: str):
    """Muestra el resultado de una ronda."""
    simbolos = {"r": "Piedra", "p": "Papel", "s": "Tijera"}

    print(f"\n--- Ronda {ronda} ---")
    print(f"IA predijo: {simbolos[jugada_ia]} ({jugada_ia})")
    print(f"Tu jugaste: {simbolos[jugada_humano]} ({jugada_humano})")

    if resultado == "victoria":
        print(">>> IA GANA <<<")
    elif resultado == "derrota":
        print(">>> Tu ganas <<<")
    else:
        print(">>> Empate <<<")


def mostrar_progreso(victorias: int, derrotas: int, empates: int, total: int):
    """Muestra el progreso actual."""
    jugadas = victorias + derrotas + empates
    restantes = total - jugadas

    if victorias + derrotas > 0:
        winrate_actual = victorias / (victorias + derrotas) * 100
        print(f"\n[Progreso: {jugadas}/{total}] "
              f"IA: {victorias}V-{derrotas}D-{empates}E "
              f"(Winrate: {winrate_actual:.1f}%) "
              f"| Quedan: {restantes}")


def evaluar(num_rondas: int = 50):
    """
    Ejecuta la evaluacion del modelo CORREGIDA.

    Args:
        num_rondas: Numero de rondas a jugar
    """
    print("="*60)
    print("   RPSAI - EVALUACION DE WINRATE CORREGIDA")
    print("="*60)
    print(f"\nSe jugaran {num_rondas} rondas contra tu modelo de IA.")
    print("Juega de forma natural, como lo harias normalmente.\n")
    print("ORDEN CORRECTO:")
    print("  1. IA predice tu PRÓXIMA jugada")
    print("  2. Tú juegas")
    print("  3. Se compara predicción vs realidad\n")

    # Intentar cargar el modelo
    try:
        ia = JugadorIA()
        if ia.modelo is None:
            print("[!] ADVERTENCIA: No se cargo ningun modelo.")
            print("[!] La IA jugara de forma ALEATORIA.")
            print("[!] Entrena tu modelo primero con: python src/modelo_mejorado_corregido.py\n")
    except Exception as e:
        print(f"[!] Error al cargar el modelo: {e}")
        print("[!] La IA jugara de forma ALEATORIA.\n")
        ia = JugadorIA()

    input("Presiona ENTER para comenzar la evaluacion...")

    victorias = 0
    derrotas = 0
    empates = 0

    for ronda in range(1, num_rondas + 1):
        # ============================================
        # ORDEN CORREGIDO:
        # 1. IA decide PRIMERO (predice tu próxima jugada)
        # 2. Humano juega DESPUÉS
        # 3. Se compara
        # ============================================

        # PASO 1: IA predice tu próxima jugada
        jugada_ia = ia.decidir_jugada()

        # PASO 2: Humano juega
        jugada_humano = leer_jugada_humano()

        # PASO 3: Registrar la jugada del humano para futuras predicciones
        ia.registrar(jugada_humano)

        # PASO 4: Determinar resultado (desde perspectiva IA)
        resultado = obtener_resultado(jugada_ia, jugada_humano)

        # Mostrar resultado
        mostrar_ronda(ronda, jugada_ia, jugada_humano, resultado)

        # Actualizar contadores
        if resultado == "victoria":
            victorias += 1
        elif resultado == "derrota":
            derrotas += 1
        else:
            empates += 1

        # Mostrar progreso
        mostrar_progreso(victorias, derrotas, empates, num_rondas)

    # Resultados finales
    print("\n" + "="*60)
    print("   RESULTADOS FINALES")
    print("="*60)

    total_decisivas = victorias + derrotas
    if total_decisivas > 0:
        winrate = victorias / total_decisivas * 100
    else:
        winrate = 0

    print(f"\nRondas jugadas: {num_rondas}")
    print(f"Victorias IA: {victorias}")
    print(f"Derrotas IA: {derrotas}")
    print(f"Empates: {empates}")
    print(f"\nWINRATE DE LA IA: {winrate:.1f}%")

    nota, bonus = obtener_nota(winrate)
    print(f"\n{'='*60}")
    print(f"   NOTA MODO 1 (Winrate): {nota}/10")
    if bonus:
        print(f"   {bonus}")
    print(f"{'='*60}")

    # Tabla de referencia
    print("\nTabla de referencia:")
    print("  33%- = 0  |  35% = 1  |  37% = 2  |  39% = 3")
    print("  40% = 4   |  42% = 5  |  44% = 6  |  46% = 7")
    print("  48% = 8   |  49% = 9  |  50%+ = 10 | 55%+ = BONUS")


def main():
    """Funcion principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Evalua el winrate de tu modelo de IA")
    parser.add_argument("-n", "--rondas", type=int, default=50,
                        help="Numero de rondas a jugar (default: 50)")
    args = parser.parse_args()

    evaluar(args.rondas)


if __name__ == "__main__":
    main()