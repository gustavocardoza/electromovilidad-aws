"""Utilidades compartidas para tests que dependen de la red vial OSM.

Los tests en modo ``road`` solo corren si existe la caché local del grafo
(``outputs/cache/drive_graph.graphml``), así la suite nunca necesita internet.
"""

import pytest

from src import street_routes

GRAPH_AVAILABLE = street_routes.GRAPH_CACHE.exists()

requires_graph = pytest.mark.skipif(
    not GRAPH_AVAILABLE, reason="sin caché de red vial OSM (drive_graph.graphml)"
)


def default_distance_mode() -> str:
    """'road' si hay red vial en caché; si no, 'haversine'."""
    return "road" if GRAPH_AVAILABLE else "haversine"
