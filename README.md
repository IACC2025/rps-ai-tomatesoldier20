[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-PyHVOwz)
# RPSAI - Piedra, Papel o Tijera con Inteligencia Artificial

**Proyecto del curso de Inteligencia Artificial - CPIFP Los Enlaces**

## Objetivo

Crear un modelo de Machine Learning capaz de predecir la siguiente jugada del oponente en el juego Piedra, Papel o Tijera, y usarlo para ganar partidas.

## Estructura del Proyecto

```
RPSAI/
├── README.md              # Este archivo
├── DATOS.md               # Documentacion de como recogiste los datos
├── requirements.txt       # Dependencias del proyecto
├── src/
│   ├── modelo.py         # TU MODELO DE IA (implementar aqui)
│   ├── evaluador.py      # Script de evaluacion del winrate
│   └── (tu codigo)       # Aqui puedes añadir mas archivos
├── data/
│   └── partidas.csv      # Tus datos de partidas
└── models/
    └── (modelos guardados)
```

## Instalacion

```bash
pip install -r requirements.txt
```

---

## Que tienes que hacer

### 1. Recolectar Datos (30% de la nota)

**Debes crear tu propio sistema para recolectar datos de partidas.**

Opciones:
- Crear un programa que permita jugar y guarde los datos
- Jugar manualmente y apuntar los resultados
- Usar una aplicacion externa y exportar los datos

**Formato minimo del CSV** (`data/partidas.csv`):

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
```

**Documenta tu proceso en `DATOS.md`**

Si capturas datos adicionales (tiempo de reaccion, timestamp, etc.) **indicalo** en DATOS.md.

### 2. Implementar tu Modelo (30% + 40% de la nota)

Edita `src/modelo.py` y completa las secciones marcadas con `TODO`:

- **Feature Engineering (30%)**: Crear caracteristicas predictivas
- **Entrenamiento (40%)**: Entrenar y seleccionar el mejor modelo

### 3. Evaluar tu IA

Ejecuta el evaluador para medir el winrate de tu modelo:

```bash
python src/evaluador.py
```

Esto jugara partidas contra tu IA y mostrara el winrate.

---

## Criterios de Evaluacion

### Modo 1: Winrate (Partidas jugadas)

| Winrate | Nota |
|---------|------|
| < 33%   | 0    |
| 35%     | 1    |
| 37%     | 2    |
| 39%     | 3    |
| 40%     | 4    |
| 42%     | 5    |
| 44%     | 6    |
| 46%     | 7    |
| 48%     | 8    |
| 49%     | 9    |
| >= 50%  | 10   |
| >= 55%  | **Fallos en examen no restan** |

**Nota**: Un modelo aleatorio tiene ~33% de winrate. Superar el 50% significa que tu IA ha aprendido patrones reales del oponente.

### Modo 2: Revision de Codigo

| Componente | Peso |
|------------|------|
| Extraccion de datos | 30% |
| Feature Engineering | 30% |
| Entrenamiento y funcionamiento del modelo | 40% |

---

## Tipos de IA y Evaluacion

### IA Especifica (para un oponente concreto)

Si tu IA esta entrenada para ganar a UNA persona especifica:
- **30 partidas** las jugara esa persona
- **20 partidas** las jugara el profesor

### IA General

Si tu IA esta entrenada para ganar a cualquier oponente:
- **50 partidas** las jugara el profesor

---

## Archivos a entregar

1. `DATOS.md` - Documentacion de como recogiste los datos
2. `src/modelo.py` - Tu modelo implementado
3. `data/partidas.csv` - Tus datos de entrenamiento
4. (Opcional) Codigo adicional de recogida de datos en `src/`

---

## Consejos

1. **Empieza simple**: Primero haz que funcione con features basicas
2. **Mas datos = mejor modelo**: Recoge al menos 100-200 rondas
3. **Prueba varios modelos**: KNN, DecisionTree, RandomForest...
4. **Analiza los errores**: La matriz de confusion te dice donde falla tu IA
5. **No hagas overfitting**: Un 95% en train pero 30% en test es malo

## Recursos

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- Material del curso (clases 05, 06, 07)
