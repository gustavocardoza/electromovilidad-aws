"""Red vial real (OpenStreetMap): matriz de distancias y rutas por calles.

Este módulo cumple dos funciones:

1. **Matriz de distancias por calles** (``road_distance_matrix_km``), que es la
   que usa el solver EVRP por defecto (``Config.distance_mode = "road"``). Cada
   nodo se "engancha" al nodo de la red más cercano y la distancia i->j es::

       d(i, j) = hav(p_i, s_i) + camino_mas_corto(s_i, s_j) + hav(s_j, p_j)

   con un Dijkstra por origen sobre la red **dirigida** (respeta sentidos de
   circulación, así que la matriz NO es simétrica). Los tramos de acceso
   (Haversine entre el punto y su nodo de red) evitan ceros falsos cuando dos
   puntos se enganchan al mismo nodo. La matriz queda en caché por instancia en
   ``outputs/cache/road_matrix_<instancia>.npz``.

2. **Reconstrucción por calles** de las rutas ya resueltas (``street_route``),
   para dibujar sobre un mapa por qué calles pasa cada vehículo. Usa la misma
   fórmula que la matriz, así que en modo ``road`` la distancia reconstruida
   coincide con la que optimizó el solver.

Es el camino más corto por longitud sobre una instantánea de OSM, no una traza
GPS real (sin tráfico ni penalización de giros).

La red vial se descarga una sola vez (requiere internet, ~90 s) y queda en
caché en ``outputs/cache/drive_graph.graphml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from haversine import haversine, haversine_vector, Unit

from . import config, data_loading

GRAPH_CACHE = config.OUTPUTS_DIR / "cache" / "drive_graph.graphml"
# Caché de matrices por calles (constante de módulo: no depende de
# config.OUTPUTS_DIR en tiempo de llamada).
ROAD_CACHE_DIR = GRAPH_CACHE.parent
# Margen alrededor de los nodos al descargar la red (grados, ~1.5 km).
BBOX_BUFFER_DEG = 0.015

# Grafo en memoria (se carga una vez por proceso).
_GRAPH = None


@dataclass
class StreetLeg:
    """Un tramo entre dos paradas consecutivas, por calles."""

    from_id: str
    to_id: str
    road_km: float
    coords: list[tuple[float, float]]  # (lat, lon) a lo largo de las calles


@dataclass
class StreetRoute:
    """Ruta completa de un vehículo reconstruida por calles."""

    vehicle: int
    legs: list[StreetLeg] = field(default_factory=list)

    @property
    def road_km(self) -> float:
        return sum(leg.road_km for leg in self.legs)

    @property
    def coords(self) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for leg in self.legs:
            out.extend(leg.coords if not out else leg.coords[1:])
        return out


def _instances_bbox() -> tuple[float, float, float, float]:
    """Bounding box (oeste, sur, este, norte) que cubre todas las instancias."""
    frames = [pd.read_csv(p) for p in config.INSTANCE_FILES.values() if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    return (
        float(df["longitude"].min()) - BBOX_BUFFER_DEG,
        float(df["latitude"].min()) - BBOX_BUFFER_DEG,
        float(df["longitude"].max()) + BBOX_BUFFER_DEG,
        float(df["latitude"].max()) + BBOX_BUFFER_DEG,
    )


def load_drive_graph(cache_path: Path = GRAPH_CACHE, refresh: bool = False):
    """Carga la red vial conducible (desde caché o descargándola de OSM).

    Se conserva solo la componente fuertemente conexa más grande, para que
    siempre exista un camino entre dos puntos respetando los sentidos de calle.
    """
    import osmnx as ox

    if cache_path.exists() and not refresh:
        return ox.load_graphml(cache_path)

    west, south, east, north = _instances_bbox()
    graph = ox.graph_from_bbox(
        bbox=(west, south, east, north), network_type="drive", simplify=True
    )
    graph = ox.truncate.largest_component(graph, strongly=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, cache_path)
    return graph


def get_drive_graph():
    """Red vial memoizada en el proceso (evita recargar el GraphML cada vez).

    Si no hay caché local y la descarga falla, lanza un error claro sugiriendo
    el modo Haversine.
    """
    global _GRAPH
    if _GRAPH is None:
        try:
            _GRAPH = load_drive_graph()
        except Exception as exc:
            if GRAPH_CACHE.exists():
                raise
            raise RuntimeError(
                "No se pudo cargar/descargar la red vial de OSM "
                f"({type(exc).__name__}: {exc}). Se necesita internet la primera "
                f"vez para crear {GRAPH_CACHE}. Usa --distance haversine "
                "(o Config.distance_mode='haversine') para resolver sin red vial."
            ) from exc
    return _GRAPH


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS))


def compute_road_distance_matrix_km(
    instance: data_loading.RoutingInstance, graph
) -> np.ndarray:
    """Matriz NxN (km) de distancias por calles, dirigida (orden all_nodes).

    ``d(i, j) = hav(p_i, s_i) + road(s_i, s_j) + hav(s_j, p_j)`` para i != j,
    diagonal 0. ``road`` sale de un Dijkstra por nodo de red de origen (peso
    ``length`` en metros). No se simetriza: las calles de un sentido hacen que
    d(i, j) != d(j, i).
    """
    import networkx as nx
    import osmnx as ox

    nodes = instance.all_nodes
    n = len(nodes)
    lats = [nd.latitude for nd in nodes]
    lons = [nd.longitude for nd in nodes]
    snaps = [int(s) for s in ox.distance.nearest_nodes(graph, X=lons, Y=lats)]

    # Tramo de acceso: punto -> nodo de red más cercano (línea recta).
    snap_coords = np.array(
        [[graph.nodes[s]["y"], graph.nodes[s]["x"]] for s in snaps], dtype=float
    )
    access_km = haversine_vector(
        np.column_stack([lats, lons]), snap_coords, unit=Unit.KILOMETERS
    )

    # Un Dijkstra por nodo de red de origen único.
    road_from: dict[int, dict[int, float]] = {}
    for s in set(snaps):
        road_from[s] = nx.single_source_dijkstra_path_length(graph, s, weight="length")

    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        lengths = road_from[snaps[i]]
        for j in range(n):
            if i == j:
                continue
            if snaps[j] not in lengths:
                raise RuntimeError(
                    f"Sin camino por calles de {nodes[i].node_id} a "
                    f"{nodes[j].node_id} (nodos OSM {snaps[i]} -> {snaps[j]})."
                )
            mat[i, j] = access_km[i] + lengths[snaps[j]] / 1000.0 + access_km[j]
    return mat


def _instance_signature(instance: data_loading.RoutingInstance):
    node_ids = np.array([nd.node_id for nd in instance.all_nodes], dtype=str)
    coords = np.array(
        [[nd.latitude, nd.longitude] for nd in instance.all_nodes], dtype=float
    )
    return node_ids, coords


def _load_cached_matrix(path: Path, node_ids: np.ndarray, coords: np.ndarray):
    """Devuelve la matriz en caché si es válida para la instancia, si no None."""
    if not path.exists():
        return None
    if GRAPH_CACHE.exists() and GRAPH_CACHE.stat().st_mtime > path.stat().st_mtime:
        return None  # la red vial es más nueva que la matriz
    try:
        with np.load(path, allow_pickle=False) as data:
            dist = data["dist_km"]
            cached_ids = data["node_ids"]
            cached_coords = data["coords"]
    except Exception:
        return None
    n = len(node_ids)
    if (
        dist.shape != (n, n)
        or cached_ids.shape != node_ids.shape
        or not np.array_equal(cached_ids, node_ids)
        or cached_coords.shape != coords.shape
        or not np.allclose(cached_coords, coords, rtol=0.0, atol=1e-9)
    ):
        return None
    return dist.astype(float)


def road_distance_matrix_km(
    instance: data_loading.RoutingInstance,
    cache_dir: Path = ROAD_CACHE_DIR,
    graph=None,
) -> np.ndarray:
    """Matriz de distancias por calles con caché por instancia.

    Se reutiliza ``<cache_dir>/road_matrix_<instancia>.npz`` si coinciden los
    ``node_id`` y coordenadas y la red vial no es más nueva que la caché; en otro
    caso se recalcula y se guarda.
    """
    node_ids, coords = _instance_signature(instance)
    path = Path(cache_dir) / f"road_matrix_{instance.name}.npz"
    cached = _load_cached_matrix(path, node_ids, coords)
    if cached is not None:
        return cached

    graph = graph if graph is not None else get_drive_graph()
    dist = compute_road_distance_matrix_km(instance, graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, dist_km=dist, node_ids=node_ids, coords=coords)
    return dist


def _edge_coords(graph, u: int, v: int) -> tuple[list[tuple[float, float]], float]:
    """Geometría (lat, lon) y largo (m) de la arista u->v más corta."""
    edges = graph.get_edge_data(u, v)
    data = min(edges.values(), key=lambda d: d.get("length", float("inf")))
    if "geometry" in data:
        coords = [(lat, lon) for lon, lat in data["geometry"].coords]
    else:
        coords = [
            (graph.nodes[u]["y"], graph.nodes[u]["x"]),
            (graph.nodes[v]["y"], graph.nodes[v]["x"]),
        ]
    return coords, float(data.get("length", 0.0))


def graph_edge_segments(graph) -> list[list[tuple[float, float]]]:
    """Geometría de todas las calles de la red, para dibujar el fondo del mapa.

    Devuelve una lista de polilíneas; cada una es una lista de ``(lat, lon)``.
    Usa la geometría real de la arista cuando existe (calles curvas) y, si no,
    la recta entre los dos nodos extremos. Pensado para pintar la red vial en
    gris claro bajo las rutas, sin depender de mosaicos externos (OSM/CARTO).
    """
    segments: list[list[tuple[float, float]]] = []
    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is not None:
            segments.append([(lat, lon) for lon, lat in geom.coords])
        else:
            segments.append(
                [
                    (graph.nodes[u]["y"], graph.nodes[u]["x"]),
                    (graph.nodes[v]["y"], graph.nodes[v]["x"]),
                ]
            )
    return segments


def street_route(graph, stops: pd.DataFrame, vehicle: int) -> StreetRoute:
    """Reconstruye por calles la ruta de un vehículo.

    ``stops`` son las filas de ``route_<instancia>.csv`` de ese vehículo, en
    orden, con columnas ``node_id``, ``latitude`` y ``longitude``.
    """
    import osmnx as ox

    stops = stops.reset_index(drop=True)
    osm_nodes = ox.distance.nearest_nodes(
        graph, X=stops["longitude"].to_list(), Y=stops["latitude"].to_list()
    )

    route = StreetRoute(vehicle=vehicle)
    for i in range(len(stops) - 1):
        a, b = stops.iloc[i], stops.iloc[i + 1]
        orig, dest = int(osm_nodes[i]), int(osm_nodes[i + 1])
        coords: list[tuple[float, float]] = [(a["latitude"], a["longitude"])]
        # Tramos de acceso/salida (misma fórmula que la matriz del solver).
        access_km = _haversine_km(
            a["latitude"], a["longitude"], graph.nodes[orig]["y"], graph.nodes[orig]["x"]
        ) + _haversine_km(
            graph.nodes[dest]["y"], graph.nodes[dest]["x"], b["latitude"], b["longitude"]
        )
        length_m = 0.0
        if orig != dest:
            path = ox.routing.shortest_path(graph, orig, dest, weight="length")
            if path is None:
                raise RuntimeError(
                    f"No hay camino por calles entre {a['node_id']} y {b['node_id']}."
                )
            for u, v in zip(path[:-1], path[1:]):
                seg, seg_len = _edge_coords(graph, u, v)
                coords.extend(seg)
                length_m += seg_len
        coords.append((b["latitude"], b["longitude"]))
        route.legs.append(
            StreetLeg(
                from_id=str(a["node_id"]),
                to_id=str(b["node_id"]),
                road_km=access_km + length_m / 1000.0,
                coords=coords,
            )
        )
    return route


def street_routes_for_instance(route_df: pd.DataFrame, graph=None) -> list[StreetRoute]:
    """Reconstruye por calles todas las rutas (una por vehículo) de un CSV."""
    graph = graph if graph is not None else load_drive_graph()
    if "vehicle" not in route_df.columns:
        route_df = route_df.assign(vehicle=0)
    out = []
    for vid, g in route_df.sort_values(["vehicle", "order"]).groupby("vehicle"):
        out.append(street_route(graph, g, int(vid)))
    return out
