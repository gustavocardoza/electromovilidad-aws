"""Test end-to-end: resuelve instancias (1 y 2 vehículos) y verifica salidas."""

import csv
import json

from src import config, solve
from tests._road import default_distance_mode, requires_graph


def _fast_cfg(fleet: int = 1):
    cfg = config.Config()
    cfg.distance_mode = default_distance_mode()
    cfg.solver.time_limit_s = 5
    cfg.solver.fleet_size = fleet
    return cfg


def test_end_to_end_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    metrics = solve.solve_instance("evaluation", cfg=_fast_cfg())
    assert metrics["feasible"], metrics.get("reason")
    assert metrics["all_visits_covered"]
    assert metrics["min_soc_frac"] >= metrics["min_soc_limit_frac"] - 1e-6
    assert (tmp_path / "route_evaluation.csv").exists()
    assert (tmp_path / "metrics_evaluation.json").exists()
    data = json.loads((tmp_path / "metrics_evaluation.json").read_text())
    assert data["instance"] == "evaluation"
    assert data["fleet_size"] == 1


def test_end_to_end_competition(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    metrics = solve.solve_instance("competition", cfg=_fast_cfg())
    assert metrics["feasible"], metrics.get("reason")
    assert metrics["all_visits_covered"]
    assert (tmp_path / "route_competition.csv").exists()
    assert (tmp_path / "metrics_competition.json").exists()


def test_end_to_end_fleet_of_two(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    cfg = _fast_cfg(fleet=2)
    cfg.solver.time_limit_s = 15
    metrics = solve.solve_instance("evaluation", cfg=cfg)
    assert metrics["feasible"], metrics.get("reason")
    assert metrics["all_visits_covered"]
    assert metrics["fleet_size"] == 2
    # Ambos vehículos usados.
    assert metrics["n_vehicles_used"] == 2
    assert len(metrics["routes"]) == 2
    assert all(rv["n_visits"] >= 1 for rv in metrics["routes"])
    # El CSV tiene la columna 'vehicle' con dos valores.
    with (tmp_path / "route_evaluation.csv").open() as f:
        rows = list(csv.DictReader(f))
    vehicles = {r["vehicle"] for r in rows}
    assert vehicles == {"0", "1"}


@requires_graph
def test_road_mode_solver_validator_consistent(tmp_path, monkeypatch):
    """En modo road, solver y validador usan la misma matriz: la energía que
    reporta OR-Tools coincide con la que resimula el validador, y los km del
    CSV suman los km de cada ruta."""
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    cfg = _fast_cfg(fleet=2)
    cfg.distance_mode = "road"
    cfg.solver.time_limit_s = 15
    metrics = solve.solve_instance("evaluation", cfg=cfg)
    assert metrics["feasible"], metrics.get("reason")
    assert metrics["distance_mode"] == "road"
    assert metrics["total_drive_km"] > 0
    assert metrics["min_soc_frac"] >= metrics["min_soc_limit_frac"] - 1e-6
    for rv, pv in zip(metrics["routes"], metrics["per_vehicle"]):
        assert abs(rv["drive_energy_kwh"] - pv["drive_energy_kwh"]) < 1e-3
        assert pv["drive_km"] == rv["drive_km"]

    with (tmp_path / "route_evaluation.csv").open() as f:
        rows = list(csv.DictReader(f))
    for rv in metrics["routes"]:
        km_csv = sum(float(r["arc_km"]) for r in rows if int(r["vehicle"]) == rv["vehicle"])
        assert abs(km_csv - rv["drive_km"]) < 1e-2
