"""Tests del modelo EVRP en OR-Tools (flota, recarga, SOC)."""

from src import config, data_loading, distance
from src.consumption_model import ConsumptionModel
from src import evrp_model
from tests._road import default_distance_mode


def _fast_cfg(cap_kwh: float | None = None, fleet: int = 1) -> config.Config:
    cfg = config.Config()
    cfg.distance_mode = default_distance_mode()
    cfg.solver.time_limit_s = 5
    cfg.solver.fleet_size = fleet
    if cap_kwh is not None:
        cfg.vehicle.battery_capacity_kwh = cap_kwh
    return cfg


def _model(cap_kwh: float) -> ConsumptionModel:
    base = ConsumptionModel.default()
    base.params.kwh_per_km = 0.33
    base.params.battery_capacity_kwh = cap_kwh
    return base


def _min_soc(res) -> float:
    return min(min(r.soc_at_nodes_kwh) for r in res.routes)


def _all_visited(res, inst) -> list[str]:
    out = []
    for r in res.routes:
        out.extend([nid for nid in r.node_ids if nid.startswith("V")])
    return out


# --- VRP base de 1 vehículo sin recargas -----------------------------------

def test_evrp_base_visits_all_once():
    inst = data_loading.load_instance("evaluation")
    model = _model(cap_kwh=200.0)  # batería enorme: no necesita recargas
    res = evrp_model.solve_evrp(inst, model, _fast_cfg(), use_rechargers=False)
    assert res.feasible
    assert res.status == "OK"
    assert len(res.routes) == 1
    route = res.routes[0]
    assert route.node_ids[0] == "BASE"
    assert route.node_ids[-1] == "BASE"
    visited = _all_visited(res, inst)
    assert sorted(visited) == sorted(v.node_id for v in inst.visits)
    assert len(visited) == 20
    assert res.total_drive_energy_kwh > 0


# --- Recarga opcional y restricción dura de SOC ----------------------------

def test_recharge_feasibility_inserts_charger():
    """Con batería pequeña, la ruta directa agota la batería: el solver debe
    insertar al menos un cargador y respetar el SOC mínimo."""
    inst = data_loading.load_instance("evaluation")
    model = _model(cap_kwh=12.0)
    cfg = _fast_cfg(cap_kwh=12.0)
    cfg.solver.time_limit_s = 15

    res = evrp_model.solve_evrp(inst, model, cfg, use_rechargers=True)
    assert res.feasible, f"status={res.status}"
    assert len(res.all_chargers_used) >= 1
    assert _min_soc(res) >= cfg.vehicle.min_soc_kwh - 1e-6


def test_soc_min_respected_without_recharge():
    """Con batería holgada, el SOC siempre está sobre el mínimo y no se recarga."""
    inst = data_loading.load_instance("evaluation")
    model = _model(cap_kwh=60.0)
    cfg = _fast_cfg(cap_kwh=60.0)
    res = evrp_model.solve_evrp(inst, model, cfg, use_rechargers=True)
    assert res.feasible
    assert _min_soc(res) >= cfg.vehicle.min_soc_kwh - 1e-6


def test_infeasible_tiny_battery_pattern():
    """Batería minúscula: el solver puede devolver NO_SOLUTION. Sin excepción."""
    inst = data_loading.load_instance("evaluation")
    model = _model(cap_kwh=1.0)
    cfg = _fast_cfg(cap_kwh=1.0)
    res = evrp_model.solve_evrp(inst, model, cfg, use_rechargers=True)
    assert res.status in {"OK", "NO_SOLUTION", "MISSING_VISITS"}


# --- Flota de N vehículos --------------------------------------------------

def test_fleet_two_vehicles_all_used_and_cover_visits():
    """Flota de 2: ambos vehículos se usan, cubren todas las visitas sin
    solaparse y respetan el SOC mínimo."""
    inst = data_loading.load_instance("evaluation")
    model = _model(cap_kwh=60.0)
    cfg = _fast_cfg(cap_kwh=60.0, fleet=2)
    cfg.solver.time_limit_s = 15

    res = evrp_model.solve_evrp(inst, model, cfg, use_rechargers=True)
    assert res.feasible, f"status={res.status}"
    assert len(res.routes) == 2
    # Ambos vehículos obligatorios: cada uno al menos una visita.
    assert all(r.n_visits >= 1 for r in res.routes)
    # Cobertura exacta: 20 visitas, sin repetir.
    visited = _all_visited(res, inst)
    assert len(visited) == 20
    assert sorted(visited) == sorted(v.node_id for v in inst.visits)
    # SOC mínimo respetado en toda la flota.
    assert _min_soc(res) >= cfg.vehicle.min_soc_kwh - 1e-6


def test_fleet_three_vehicles_all_used():
    inst = data_loading.load_instance("evaluation")
    model = _model(cap_kwh=60.0)
    cfg = _fast_cfg(cap_kwh=60.0, fleet=3)
    cfg.solver.time_limit_s = 15
    res = evrp_model.solve_evrp(inst, model, cfg, use_rechargers=True)
    assert res.feasible, f"status={res.status}"
    assert len(res.routes) == 3
    assert all(r.n_visits >= 1 for r in res.routes)
    visited = _all_visited(res, inst)
    assert len(visited) == 20


# --- Matriz de distancias explícita (fuente única de verdad) ---------------

def test_explicit_dist_km_energy_consistent():
    """Con dist_km explícita, la energía reportada por el solver coincide con
    la recalculada desde esa misma matriz a lo largo de las rutas."""
    inst = data_loading.load_instance("evaluation")
    model = _model(cap_kwh=60.0)
    cfg = _fast_cfg(cap_kwh=60.0, fleet=2)
    dist = distance.build_distance_matrix(inst, mode="haversine")
    energy = distance.energy_matrix_kwh(inst, model, dist)

    res = evrp_model.solve_evrp(inst, model, cfg, use_rechargers=True, dist_km=dist)
    assert res.feasible, f"status={res.status}"
    recomputed = sum(
        float(energy[a][b])
        for r in res.routes
        for a, b in zip(r.node_indices[:-1], r.node_indices[1:])
    )
    assert abs(res.total_drive_energy_kwh - recomputed) < 1e-3
