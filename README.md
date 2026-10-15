# EVRP — Electric Vehicle Routing Problem

Resolución de un **Electric Vehicle Routing Problem (EVRP)** sobre dos
instancias de nodos de ruteo. Un vehículo eléctrico parte de una base, debe
visitar 20 nodos (V1–V20) y regresar, pudiendo recargar en una red de estaciones
(C1–C62, cada una con potencia y conector distintos).

**Objetivo:** minimizar la energía total de conducción, con la restricción dura
de que el estado de carga (SOC) **nunca** baje de un mínimo de seguridad (nunca
se queda sin batería).

Soporta una **flota de N vehículos homogéneos**: todos parten de la base, se
reparten las visitas (cada visita la atiende un solo vehículo) y el objetivo es
minimizar la **energía total** de la flota. Por defecto cada vehículo debe
atender al menos una visita.

## Enfoque

Arquitectura híbrida en tres etapas, sobre **distancias reales por calles**:

0. **Matriz de distancias por red vial** (por defecto): cada nodo se engancha a
   la intersección más cercana de la red conducible de OpenStreetMap y la
   distancia i→j es el camino más corto **dirigido** (respeta calles de un
   sentido, un Dijkstra por origen), más los tramos de acceso punto↔red en línea
   recta. La matriz 81×81 es asimétrica y la usan por igual el solver, la
   recarga y el validador. Alternativa sin red vial: `--distance haversine`.
1. **Modelo de consumo** calibrado con telemetría real de EVs (`eVED_EV.csv`):
   estima el consumo en kWh/km y la capacidad de batería.
2. **Optimización con OR-Tools**: decide cómo repartir las visitas entre los
   vehículos, el orden de cada ruta y qué cargadores usar. La batería se modela
   como una **dimensión de SOC restante por vehículo** (empieza en la capacidad,
   decrece por arco con el consumo, puede subir en los cargadores). La cota
   inferior del SOC es una **restricción dura**: el solver nunca entrega una
   ruta que agote la batería. Que cada vehículo se use se garantiza con una
   dimensión que cuenta visitas (cumul final ≥ 1 por vehículo).
3. **Post-proceso de recarga parcial**: para la secuencia de cargadores de cada
   ruta, calcula de forma determinista cuánta energía reponer en cada parada
   (solo lo necesario para el siguiente tramo + margen) y el tiempo de recarga
   según la `max_power_kw` de esa estación concreta.

Un **validador independiente** resimula el SOC paso a paso para confirmar la
factibilidad y detectar cualquier desajuste de redondeo/escala del solver.

> Por qué híbrido: OR-Tools garantiza SOC > mínimo como restricción dura y
> minimiza la energía de conducción de forma escalable. La relación no lineal
> *tiempo = energía / potencia* (distinta por estación) no se modela bien dentro
> de OR-Tools, así que se resuelve en el post-proceso determinista.

## Estructura

```
src/
  config.py             parámetros (batería, SOC mín, conector, solver)
  data_loading.py       carga de nodos de ruteo y telemetría EV
  consumption_model.py  calibración del consumo (kWh/km) y capacidad
  distance.py           matriz de distancias (road/haversine) + energía por arco
  street_routes.py      red vial OSM: matriz por calles (caché) y rutas por calles
  evrp_model.py         modelo OR-Tools (dimensión SOC restante + recargas)
  recharge.py           recarga parcial realista por potencia de estación
  validate.py           validador independiente de SOC + métricas
  solve.py              CLI end-to-end
notebooks/
  01_consumption_model.ipynb    análisis del modelo de consumo
  02_routes_visualization.ipynb mapas de rutas y curvas de SOC
  03_routes_matplotlib.ipynb    rutas/SOC en matplotlib + ruta por calles
outputs/              rutas (CSV) y métricas (JSON) generadas
outputs/cache/        red vial OSM (graphml) y matrices por calles (npz)
tests/                suite de pruebas (pytest)
```

## Instalación

Requiere Python 3.10+ (probado en 3.14).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

La primera ejecución en modo `road` descarga la red vial de OSM (~90 s, requiere
internet) y la guarda en `outputs/cache/drive_graph.graphml`; la matriz por
calles de cada instancia queda en `outputs/cache/road_matrix_<instancia>.npz` y
se recalcula sola si cambian los nodos o la red. Sin internet ni caché, usa
`--distance haversine`.

## Uso

Resolver ambas instancias con un vehículo y generar salidas:

```bash
python -m src.solve --instance all
```

Resolver con una **flota de 2 vehículos**:

```bash
python -m src.solve --instance all --vehicles 2
```

Resolver una sola instancia con más tiempo de cómputo:

```bash
python -m src.solve --instance competition --time-limit 60
```

Forzar recargas (p. ej. con una batería más pequeña) para ver el post-proceso:

```bash
python -m src.solve --instance evaluation --battery 14
```

Opciones principales:

| Opción          | Descripción                                             |
|-----------------|---------------------------------------------------------|
| `--instance`    | `evaluation`, `competition` o `all`                     |
| `--vehicles`    | Número de vehículos de la flota (por defecto 1)         |
| `--time-limit`  | Límite de cómputo de OR-Tools en segundos               |
| `--battery`     | Capacidad de batería en kWh (sobreescribe la estimada)  |
| `--min-soc`     | SOC mínimo como fracción (0–1), por defecto 0.10        |
| `--no-recharge` | Resolver sin considerar cargadores                      |
| `--distance`    | `road` (red vial OSM, por defecto) o `haversine`        |
| `--train`       | Reentrenar el modelo de consumo desde la telemetría     |

### Salidas

Por cada instancia se generan en `outputs/`:

- `route_<instancia>.csv`: rutas ordenadas con una columna `vehicle` (qué
  vehículo hace cada tramo), tipo de nodo, SOC al llegar, energía y km por arco
  (`arc_energy_kwh`, `arc_km`), energía acumulada y acciones de recarga
  (energía repuesta, potencia, tiempo).
- `metrics_<instancia>.json`: métricas de flota (factibilidad, vehículos usados,
  visitas cubiertas, energía total de conducción, `total_drive_km`, nº de
  recargas, energía/tiempo de recarga, SOC mínimo, `distance_mode`) más un
  desglose `routes` por vehículo (con `drive_km`).

### Visualización

Tras generar las salidas:

```bash
jupyter notebook notebooks/02_routes_visualization.ipynb
```

Muestra las rutas sobre un mapa (base, visitas, cargadores usados) y la curva de
SOC a lo largo de la ruta, con la línea del SOC mínimo.

## Resultados

Con la capacidad estimada desde la telemetría (~32.7 kWh) y distancias por
calles (límite de 30 s), **ambas instancias son factibles sin recargas**:

| Instancia   | Flota | Energía (kWh) | Distancia (km) | SOC mín. |
|-------------|-------|---------------|----------------|----------|
| evaluation  | 1     | 24.98         | 75.60          | 23.7%    |
| competition | 1     | 25.02         | 75.72          | 23.6%    |
| evaluation  | 2     | 25.74         | 77.92          | 25.9%    |
| competition | 2     | 25.27         | 76.48          | 25.6%    |

Con Haversine las mismas rutas salían en ~18 kWh (~54–56 km): la línea recta
subestima ~1.24× la distancia por calles (mediana). Las recargas se activan
automáticamente si la batería no alcanza (p. ej. `--battery 14` inserta paradas
en cargadores).

Con flota de más de un vehículo y el objetivo de **minimizar energía total**, el
reparto óptimo tiende a ser desigual: un vehículo asume casi todas las visitas
en una ruta eficiente y los demás atienden las visitas más cercanas a la base
(coste mínimo), porque dividir en rutas parejas añadiría kilómetros de ida y
vuelta a la base. Si prefieres rutas balanceadas, haría falta un objetivo
distinto (p. ej. límite de visitas por vehículo o minimizar la ruta más larga).

## Supuestos y limitaciones

- **Distancias por red vial OSM.** Camino más corto por **longitud** (no por
  tiempo) sobre una instantánea de OpenStreetMap: sin tráfico ni penalización
  de giros. Los tramos punto↔intersección más cercana se suman en línea recta.
  Es la forma consistente con el modelo de consumo, cuyo kWh/km se calibró con
  distancia recorrida por calles (GPS). `--distance haversine` queda como
  alternativa (subestima la distancia y por tanto la energía).
- **Consumo kWh/km único.** El modelo de consumo incluye accesorios (A/C,
  calefacción) y ralentí, no solo tracción, por eso el valor (~0.33 kWh/km)
  supera el rango de solo-tracción (0.12–0.30). Las instancias no traen
  velocidad ni elevación por nodo, así que no se aplica un término de
  pendiente/velocidad por arco (sí se analiza en el notebook de consumo).
- **Capacidad de batería estimada** desde `energía / ΔSOC` de la telemetría
  (~32.7 kWh). Se puede fijar con `--battery`.
- **Conector del EV**: por defecto CCS/J1772. Un cargador se considera
  compatible si comparte conector; es configurable en `config.py`.
- **SOC inicial 100%, SOC mínimo 10%** (parametrizable).
- **Flota homogénea**: todos los vehículos comparten batería, consumo y
  conector. El objetivo es energía total; cada vehículo debe hacer al menos una
  visita (configurable en `config.py`).
- **Recarga parcial**: se repone lo necesario para el siguiente tramo + margen,
  nunca por encima del 95% de SOC.

## Tests

```bash
pytest
```
