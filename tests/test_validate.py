"""Tests del validador independiente (Task 7)."""

from src import config, data_loading, recharge, validate
from src.consumption_model import ConsumptionModel
from src.distance import distance_matrix_km, energy_matrix_kwh


def _setup(cap_kwh=60.0, kwh_per_km=0.33):
    inst = data_loading.load_instance("evaluation")
    model = ConsumptionModel.default()
    model.params.kwh_per_km = kwh_per_km
    d = distance_matrix_km(inst)
    e = energy_matrix_kwh(inst, model, d)
    cfg = config.Config()
    cfg.vehicle.battery_capacity_kwh = cap_kwh
    return inst, e, cfg


def test_validator_accepts_soc_feasible_route():
    """Una ruta que cubre TODAS las visitas con batería holgada es factible y
    nunca baja del SOC mínimo."""
    inst, e, cfg = _setup(cap_kwh=200.0)  # batería enorme: sin recargas
    # Ruta que cubre las 20 visitas en orden V1..V20.
    ids = ["BASE"] + [v.node_id for v in inst.visits] + ["BASE"]
    id_to_global = {node.node_id: i for i, node in enumerate(inst.all_nodes)}
    idx = [id_to_global[x] for x in ids]
    plan = recharge.plan_recharge(ids, idx, inst, e, cfg)
    metrics = validate.validate_route(inst, ids, idx, e, plan, cfg)
    assert metrics.feasible, metrics.reason
    assert metrics.all_visits_covered
    assert metrics.min_soc_kwh >= cfg.vehicle.min_soc_kwh - 1e-6
    assert metrics.n_recharges == 0


def test_validator_rejects_infeasible_route():
    inst, e, cfg = _setup(cap_kwh=2.0, kwh_per_km=0.5)
    # Batería diminuta, ruta larga sin cargadores: debe agotar la batería.
    ids = ["BASE", "V1", "V2", "V3", "V4", "BASE"]
    id_to_global = {node.node_id: i for i, node in enumerate(inst.all_nodes)}
    idx = [id_to_global[x] for x in ids]
    # Sin plan de recarga (sin cargadores en la ruta).
    plan = recharge.plan_recharge(ids, idx, inst, e, cfg)
    metrics = validate.validate_route(inst, ids, idx, e, plan, cfg)
    assert not metrics.feasible


def test_validator_counts_visits():
    inst, e, cfg = _setup()
    ids = ["BASE", "V1", "V2", "BASE"]
    idx = [0, 1, 2, 0]
    plan = recharge.plan_recharge(ids, idx, inst, e, cfg)
    metrics = validate.validate_route(inst, ids, idx, e, plan, cfg)
    assert metrics.n_visits_covered == 2
    assert metrics.n_visits_required == 20
    assert not metrics.all_visits_covered  # solo 2 de 20
