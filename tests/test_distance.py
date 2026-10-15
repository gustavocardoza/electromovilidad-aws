"""Tests de matrices de distancia y energía (Task 3).

Los cuatro primeros tests usan ``distance_matrix_km`` = Haversine (explícito).
Los tests de red vial se saltan si no hay caché local del grafo OSM.
"""

import dataclasses
import os

import numpy as np
import pytest

from src import data_loading, distance, street_routes
from src.consumption_model import ConsumptionModel
from tests._road import requires_graph


def test_distance_matrix_symmetric_zero_diag():
    inst = data_loading.load_instance("evaluation")
    d = distance.distance_matrix_km(inst)
    n = len(inst.all_nodes)
    assert d.shape == (n, n)
    # Diagonal cero.
    assert np.allclose(np.diag(d), 0.0)
    # Simétrica.
    assert np.allclose(d, d.T)
    # Todas las distancias no negativas.
    assert np.all(d >= 0)


def test_known_arc_distance_plausible():
    inst = data_loading.load_instance("evaluation")
    d = distance.distance_matrix_km(inst)
    # Base -> V1 debe ser una distancia urbana de pocos km (Ann Arbor).
    base_to_v1 = d[0, 1]
    assert 0.5 < base_to_v1 < 20.0


def test_energy_matrix_nonneg_proportional():
    inst = data_loading.load_instance("evaluation")
    model = ConsumptionModel.default()
    d = distance.distance_matrix_km(inst)
    e = distance.energy_matrix_kwh(inst, model, d)
    assert np.all(e >= 0)
    assert np.allclose(np.diag(e), 0.0)
    # Proporcional a la distancia: e = kwh_per_km * d (en arcos positivos).
    mask = ~np.eye(d.shape[0], dtype=bool)
    assert np.allclose(e[mask], model.kwh_per_km * d[mask])


def test_energy_matrix_scaled_integers():
    inst = data_loading.load_instance("evaluation")
    model = ConsumptionModel.default()
    e = distance.energy_matrix_kwh(inst, model)
    s = distance.energy_matrix_scaled(e)
    assert s.dtype == np.int64
    assert np.all(s >= 0)
    assert np.all(np.diag(s) == 0)


# --- Punto único de entrada: build_distance_matrix -------------------------

def test_build_distance_matrix_haversine_symmetric():
    inst = data_loading.load_instance("evaluation")
    d = distance.build_distance_matrix(inst, mode="haversine")
    assert np.allclose(d, d.T)
    assert np.array_equal(d, distance.distance_matrix_km(inst))


def test_build_distance_matrix_invalid_mode():
    inst = data_loading.load_instance("evaluation")
    with pytest.raises(ValueError):
        distance.build_distance_matrix(inst, mode="manhattan")


# --- Modo red vial (requiere caché local del grafo OSM) ---------------------

def _offdiag():
    return ~np.eye(81, dtype=bool)


@requires_graph
@pytest.mark.parametrize("name", ["evaluation", "competition"])
def test_road_matrix_shape_diag_positive(name):
    inst = data_loading.load_instance(name)
    d = distance.build_distance_matrix(inst, mode="road")
    h = distance.distance_matrix_km(inst)
    n = len(inst.all_nodes)
    assert n == 81
    assert d.shape == (n, n)
    assert np.all(np.diag(d) == 0.0)
    m = _offdiag()
    assert np.all(np.isfinite(d))
    # > 0 en todo par con coordenadas distintas (D11).
    distinct = m & (h > 1e-9)
    assert np.all(d[distinct] > 0)


@requires_graph
@pytest.mark.parametrize("name", ["evaluation", "competition"])
def test_road_at_least_haversine(name):
    inst = data_loading.load_instance(name)
    d = distance.build_distance_matrix(inst, mode="road")
    h = distance.distance_matrix_km(inst)
    m = _offdiag()
    assert np.all(d[m] >= 0.99 * h[m])


@requires_graph
def test_road_matrix_not_forced_symmetric():
    """Calles de un sentido: la matriz por calles NO se simetriza."""
    inst = data_loading.load_instance("evaluation")
    d = distance.build_distance_matrix(inst, mode="road")
    assert not np.allclose(d, d.T)


def _raising_stub(*args, **kwargs):
    raise AssertionError("no debía recalcular: la caché era válida")


@requires_graph
def test_road_matrix_cache_reused(tmp_path, monkeypatch):
    inst = data_loading.load_instance("evaluation")
    d1 = street_routes.road_distance_matrix_km(inst, cache_dir=tmp_path)
    assert (tmp_path / "road_matrix_evaluation.npz").exists()
    monkeypatch.setattr(street_routes, "compute_road_distance_matrix_km", _raising_stub)
    d2 = street_routes.road_distance_matrix_km(inst, cache_dir=tmp_path)
    assert np.array_equal(d1, d2)


@requires_graph
def test_road_matrix_cache_invalidated(tmp_path, monkeypatch):
    """Si cambian las coordenadas o la red vial es más nueva, se recalcula."""
    inst = data_loading.load_instance("evaluation")
    street_routes.road_distance_matrix_km(inst, cache_dir=tmp_path)
    cache_file = tmp_path / "road_matrix_evaluation.npz"

    calls = []

    def _stub(instance, graph):
        calls.append(instance)
        return np.zeros((81, 81))

    monkeypatch.setattr(street_routes, "compute_road_distance_matrix_km", _stub)

    # 1) Coordenada de una visita alterada -> recalcula.
    moved = dataclasses.replace(inst.visits[0], latitude=inst.visits[0].latitude + 1e-3)
    inst_moved = data_loading.RoutingInstance(
        name=inst.name, base=inst.base, visits=[moved, *inst.visits[1:]],
        chargers=inst.chargers,
    )
    street_routes.road_distance_matrix_km(inst_moved, cache_dir=tmp_path)
    assert len(calls) == 1

    # 2) Red vial más nueva que la caché -> recalcula.
    graph_mtime = street_routes.GRAPH_CACHE.stat().st_mtime
    os.utime(cache_file, (graph_mtime - 10, graph_mtime - 10))
    street_routes.road_distance_matrix_km(inst_moved, cache_dir=tmp_path)
    assert len(calls) == 2
