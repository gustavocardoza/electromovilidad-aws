"""Tests del traductor web <-> solver (web_api)."""

import pytest

from src import web_api, data_loading
from src.evrp_model import FleetResult, VehicleRoute
from tests._road import default_distance_mode


# --- Traducción de entrada (JSON -> RoutingInstance) -----------------------

def test_instance_from_request_basic():
    payload = {
        "base": {"node_id": "B1", "lat": 42.277, "lon": -83.763},
        "visits": [{"node_id": "V1", "lat": 42.277, "lon": -83.731}],
        "chargers": [{"node_id": "C1", "lat": 42.241, "lon": -83.743,
                      "connector_types": ["CCS", "NACS"], "max_power_kw": 400}],
        "fleet_size": 2,
    }
    inst, fleet = web_api.instance_from_request(payload)
    assert isinstance(inst, data_loading.RoutingInstance)
    assert inst.base.node_id == "B1" and inst.base.is_base
    assert [v.node_id for v in inst.visits] == ["V1"]
    assert inst.chargers[0].node_id == "C1"
    assert inst.chargers[0].max_power_kw == 400.0
    assert inst.chargers[0].connector_types == ("CCS", "NACS")
    assert fleet == 2


def test_fleet_size_defaults_to_one():
    payload = {"base": {"node_id": "B1", "lat": 42.277, "lon": -83.763}, "visits": []}
    _, fleet = web_api.instance_from_request(payload)
    assert fleet == 1


def test_connectors_accept_string():
    inst, _ = web_api.instance_from_request({
        "base": {"node_id": "B1", "lat": 42.277, "lon": -83.763},
        "chargers": [{"node_id": "C1", "lat": 42.24, "lon": -83.74,
                      "connector_types": "CCS; NACS", "max_power_kw": 50}],
    })
    assert inst.chargers[0].connector_types == ("CCS", "NACS")


def test_missing_base_raises():
    with pytest.raises(ValueError):
        web_api.instance_from_request({"visits": []})


def test_malformed_node_raises():
    with pytest.raises(ValueError):
        web_api.instance_from_request({"base": {"node_id": "B1", "lat": 42.0}})  # falta lon


def test_fleet_size_zero_raises():
    with pytest.raises(ValueError):
        web_api.instance_from_request({
            "base": {"node_id": "B1", "lat": 42.277, "lon": -83.763},
            "fleet_size": 0,
        })


# --- Traducción de salida (FleetResult -> JSON) ----------------------------

def _instance():
    base = data_loading.Node("B1", "base", 42.277, -83.763)
    v1 = data_loading.Node("V1", "visit", 42.277, -83.731)
    c1 = data_loading.Node("C1", "charger", 42.241, -83.743,
                           max_power_kw=400.0, connector_types=("CCS",))
    return data_loading.RoutingInstance("web-test", base, [v1], [c1])


def test_response_envelope_and_format():
    inst = _instance()
    result = FleetResult(
        routes=[VehicleRoute(0, ["B1", "V1", "B1"], [0, 1, 0], 1.0, [], [33, 32, 31], 1)],
        total_drive_energy_kwh=1.0, feasible=True, status="OK",
    )
    resp = web_api.response_from_result(inst, 1, result, persistence="dynamodb")

    # Sobre generado por el traductor.
    assert resp["status"] == "completed"
    assert resp["persistence"] == "dynamodb"
    assert "request_id" in resp and "created_at" in resp
    # Estructura interna.
    r = resp["result"]
    assert r["status"] == "completed"
    assert r["summary"] == {
        "base_id": "B1", "visit_count": 1, "charger_count": 1,
        "fleet_size": 1, "visit_ids": ["V1"], "charger_ids": ["C1"],
    }
    # Formato de rutas: vehicle_id + stops.
    assert r["result"]["routes"] == [{"vehicle_id": "vehicle-1", "stops": ["B1", "V1", "B1"]}]


def test_response_no_solution():
    inst = _instance()
    result = FleetResult(routes=[], feasible=False, status="NO_SOLUTION")
    resp = web_api.response_from_result(inst, 2, result)
    assert resp["status"] == "completed"           # el sobre siempre completa
    assert resp["result"]["status"] == "no_solution"
    assert resp["result"]["result"]["routes"] == []


# --- Extremo a extremo (requiere grafo OSM para modo road) -----------------

def test_solve_request_end_to_end():
    payload = {
        "base": {"node_id": "B1", "lat": 42.277, "lon": -83.763},
        "visits": [
            {"node_id": "V1", "lat": 42.277, "lon": -83.731},
            {"node_id": "V2", "lat": 42.323, "lon": -83.691},
            {"node_id": "V3", "lat": 42.313, "lon": -83.800},
            {"node_id": "V4", "lat": 42.269, "lon": -83.676},
        ],
        "chargers": [{"node_id": "C1", "lat": 42.241, "lon": -83.743,
                      "connector_types": ["CCS"], "max_power_kw": 400}],
        "fleet_size": 2,
    }
    resp = web_api.solve_request(
        payload, config_overrides={"distance_mode": default_distance_mode(),
                                   "time_limit_s": 10},
    )
    assert resp["status"] == "completed"
    assert resp["result"]["status"] == "completed"
    routes = resp["result"]["result"]["routes"]
    assert len(routes) == 2
    # Todas las paradas empiezan y terminan en la base.
    for rt in routes:
        assert rt["stops"][0] == "B1" and rt["stops"][-1] == "B1"
    # Cobertura: las 4 visitas aparecen entre las rutas.
    visited = {s for rt in routes for s in rt["stops"] if s.startswith("V")}
    assert visited == {"V1", "V2", "V3", "V4"}


def test_solve_request_incompatible_gives_no_solution():
    """1 visita con flota de 2 (todos obligatorios) -> no hay solución."""
    payload = {
        "base": {"node_id": "B1", "lat": 42.277, "lon": -83.763},
        "visits": [{"node_id": "V1", "lat": 42.277, "lon": -83.731}],
        "chargers": [],
        "fleet_size": 2,
    }
    resp = web_api.solve_request(
        payload, config_overrides={"distance_mode": default_distance_mode(),
                                   "time_limit_s": 5},
    )
    assert resp["result"]["status"] == "no_solution"
    assert resp["result"]["result"]["routes"] == []
