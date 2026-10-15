"""Matrices de distancia y energía por arco.

``build_distance_matrix`` es el punto único de entrada para la matriz de
distancias (km) que usan el solver, el planificador de recarga y el validador:

* ``road`` (por defecto): camino más corto por la red vial dirigida de
  OpenStreetMap (ver ``street_routes.road_distance_matrix_km``). Es asimétrica
  (calles de un sentido) y es la forma consistente con la calibración kWh/km,
  que se hizo con distancia recorrida por calles.
* ``haversine``: línea recta (alternativa sin red vial). Subestima la distancia
  real; es la única matriz que se simetriza.

Con el modelo de consumo se obtiene la matriz de energía (kWh) por arco y, para
OR-Tools, su versión entera escalada (ver ``config.ENERGY_SCALE``).
"""

from __future__ import annotations

import argparse

import numpy as np
from haversine import haversine_vector, Unit

from . import config
from . import data_loading
from .consumption_model import ConsumptionModel


def distance_matrix_km(instance: data_loading.RoutingInstance) -> np.ndarray:
    """Matriz NxN de distancias Haversine en km entre todos los nodos.

    El orden de filas/columnas sigue ``instance.all_nodes``
    (base, visitas, cargadores).
    """
    nodes = instance.all_nodes
    coords = np.array([[n.latitude, n.longitude] for n in nodes], dtype=float)
    n = len(nodes)
    mat = np.zeros((n, n), dtype=float)
    # haversine_vector hace broadcasting de un punto contra el resto.
    for i in range(n):
        origins = np.repeat(coords[i : i + 1], n, axis=0)
        mat[i] = haversine_vector(origins, coords, unit=Unit.KILOMETERS)
    # Fuerza diagonal exactamente cero y simetría numérica.
    np.fill_diagonal(mat, 0.0)
    mat = 0.5 * (mat + mat.T)
    return mat


def build_distance_matrix(
    instance: data_loading.RoutingInstance,
    cfg: config.Config | None = None,
    mode: str | None = None,
) -> np.ndarray:
    """Matriz NxN de distancias (km) según el modo configurado.

    ``mode`` (si se indica) tiene prioridad sobre ``cfg.distance_mode``. El orden
    de filas/columnas sigue ``instance.all_nodes``. En modo ``road`` la matriz es
    dirigida: ``d[i, j]`` es la distancia de i a j y no se simetriza.
    """
    if mode is None:
        mode = (cfg or config.Config()).distance_mode
    if mode not in config.DISTANCE_MODES:
        raise ValueError(
            f"Modo de distancia desconocido: {mode!r}. Opciones: {config.DISTANCE_MODES}"
        )
    if mode == "haversine":
        return distance_matrix_km(instance)
    from . import street_routes  # import diferido: osmnx solo si se usa 'road'

    return street_routes.road_distance_matrix_km(instance)


def energy_matrix_kwh(
    instance: data_loading.RoutingInstance,
    model: ConsumptionModel,
    dist_km: np.ndarray | None = None,
) -> np.ndarray:
    """Matriz NxN de energía de conducción (kWh) por arco."""
    if dist_km is None:
        dist_km = distance_matrix_km(instance)
    energy = model.energy(dist_km)
    energy = np.asarray(energy, dtype=float)
    np.fill_diagonal(energy, 0.0)
    return energy


def energy_matrix_scaled(
    energy_kwh: np.ndarray, scale: int = config.ENERGY_SCALE
) -> np.ndarray:
    """Versión entera escalada de la matriz de energía para OR-Tools."""
    scaled = np.rint(energy_kwh * scale).astype(np.int64)
    np.fill_diagonal(scaled, 0)
    return scaled


def _main() -> None:
    parser = argparse.ArgumentParser(description="Estadísticas de las matrices.")
    parser.add_argument(
        "--instance", default="evaluation", choices=sorted(config.INSTANCE_FILES)
    )
    parser.add_argument(
        "--distance", default="road", choices=config.DISTANCE_MODES,
        help="Distancias por red vial OSM ('road') o línea recta ('haversine').",
    )
    args = parser.parse_args()

    inst = data_loading.load_instance(args.instance)
    try:
        model = ConsumptionModel.load()
    except FileNotFoundError:
        print("(No hay modelo entrenado; usando valores por defecto.)")
        model = ConsumptionModel.default()

    d = build_distance_matrix(inst, mode=args.distance)
    e = energy_matrix_kwh(inst, model, d)

    # Estadísticas sobre arcos fuera de la diagonal.
    mask = ~np.eye(d.shape[0], dtype=bool)
    dv = d[mask]
    ev = e[mask]
    print(f"Instancia '{args.instance}': {d.shape[0]} nodos  |  distancias: {args.distance}")
    print(f"  Matriz simétrica: {np.allclose(d, d.T)}")
    print(f"  kWh/km del modelo: {model.kwh_per_km:.4f}")
    print(
        f"  Distancia arco (km):  min={dv.min():.3f}  "
        f"mediana={np.median(dv):.3f}  max={dv.max():.3f}"
    )
    print(
        f"  Energía arco (kWh):   min={ev.min():.3f}  "
        f"mediana={np.median(ev):.3f}  max={ev.max():.3f}"
    )
    print(f"  Capacidad útil batería: {config.DEFAULT_CONFIG.vehicle.usable_capacity_kwh:.1f} kWh")


if __name__ == "__main__":
    _main()
