# Documentacion de Recogida de Datos

**Alumno:** Gabriel Orlando Cruz Parraga

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [ ] **Programa propio**: Cree un programa para jugar y guardar datos
- [x] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
Realicé 100 partidas con el compañero Petrut Cosmin Stancu; y trás esas partidas, le solicité los datos que había recolectado
de Cosmin a Diego Sanchez. 
```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [x] `resultado` - victoria/derrota/empate
- [x] Otro: Probabilidades (Piedra, Papel, Tijera), Resultado de partida (Cosmin),última jugada (Cosmin), Patrones cada 5 jugadas, Comportamientos (tras ganar, perder o empatar), Entropía. 

### Descripcion de datos adicionales:

```
- Las probabilidades son la veces de que en el dataset, el oponente a recurrido a una jugada en concreto, que es la base para intentar predecir la siguiente jugada.
- Las Frecuencias (no está en el dataset pero se mide en base a features que luego se implementan en la IA) sirve para detectar un número mayor de jugadas en base al juego actual.
- Los patrones sirven para verificar si el jugador opta por crear inconscientemente un patrón de juego a pesar de ser "totalmente aleatorio".
- El comportamiento tras el resultado de una partida (ya sea ganada, perdido o empatada) afecata a cada jugador, si tiende a cambiar o a permanecer igual. Si cambia, puede
  pretender a predecir lo que has jugado, o sacar lo que tu has sacado para tal vez asegurar un empate y trazar una estrategia.
- Para finalizar, la entropía es para medir que tan consciente de la partida y de lo que estaba sacando el oponente es. Si esta sacando de forma aleatoria, dejando a la suerte
  que le haga el trabajo o no (aleatoriedad). Cuanto mas entropico se una persona, mas aleatoria es.


```

---

## Estadisticas del dataset

- **Total de rondas:** 200 rondas
- **Número de sesiones/partidas:** 2 partidas
- **Contra cuantas personas diferentes:** 1 sola persona

### Tipo de IA:

- [x] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: Petrut Cosmin Stancu
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
