"""Tests del post-proceso de recarga (Task 6)."""

import numpy as np
import pytest

from src import config, data_loading, recharge


def test_recharge_time_formula():
    # 10 kWh a 100 kW con eficiencia 1.0 => 0.1 h.
    assert recharge.recharge_time_hours(10.0, 100.0, efficiency=1.0) == pytest.approx(0.1)
    # Con eficiencia 0.9, toma más tiempo.
    t = recharge.recharge_time_hours(10.0, 100.0, efficiency=0.9)
    assert t == pytest.approx(10.0 / 90.0)


def test_higher_power_less_time():
    t_slow = recharge.recharge_time_hours(10.0, 10.0)
    t_fast = recharge.recharge_time_hours(10.0, 100.0)
    assert t_fast < t_slow


def test_zero_power_raises():
    with pytest.raises(ValueError):
        recharge.recharge_time_hours(5.0, 0.0)


def test_plan_recharge_no_charger_feasible():
    """Ruta sin cargadores y batería holgada: plan factible, sin paradas."""
    inst = data_loading.load_instance("evaluation")
    from src.distance import distance_matrix_km, energy_matrix_kwh
    from src.consumption_model import ConsumptionModel

    model = ConsumptionModel.default()
    d = distance_matrix_km(inst)
    e = energy_matrix_kwh(inst, model, d)

    # Ruta: BASE -> V1 -> V2 -> BASE (índices globales 0,1,2,0).
    ids = ["BASE", "V1", "V2", "BASE"]
    idx = [0, 1, 2, 0]
    cfg = config.Config()
    cfg.vehicle.battery_capacity_kwh = 60.0
    plan = recharge.plan_recharge(ids, idx, inst, e, cfg)
    assert plan.feasible
    assert plan.stops == []
    assert plan.total_recharge_energy_kwh == 0.0


def test_plan_recharge_partial_charge():
    """Una parada de recarga repone energía parcial y calcula tiempo por potencia."""
    inst = data_loading.load_instance("evaluation")
    from src.distance import distance_matrix_km, energy_matrix_kwh
    from src.consumption_model import ConsumptionModel

    model = ConsumptionModel.default()
    model.params.kwh_per_km = 0.33
    d = distance_matrix_km(inst)
    e = energy_matrix_kwh(inst, model, d)

    # Encuentra el índice global de un cargador DC rápido (C1, 400 kW).
    id_to_global = {node.node_id: i for i, node in enumerate(inst.all_nodes)}
    c1 = id_to_global["C1"]
    v1 = id_to_global["V1"]
    v2 = id_to_global["V2"]

    # Ruta que pasa por el cargador: BASE -> V1 -> C1 -> V2 -> BASE.
    ids = ["BASE", "V1", "C1", "V2", "BASE"]
    idx = [0, v1, c1, v2, 0]

    cfg = config.Config()
    cfg.vehicle.battery_capacity_kwh = 12.0  # batería pequeña -> fuerza recarga
    cfg.vehicle.initial_soc_frac = 0.5       # arranca a media carga

    plan = recharge.plan_recharge(ids, idx, inst, e, cfg)
    # El plan debe ser factible y haber recargado algo en C1.
    assert plan.feasible, plan.reason
    assert len(plan.stops) >= 1
    stop = plan.stops[0]
    assert stop.node_id == "C1"
    assert stop.power_kw == 400.0
    assert stop.energy_added_kwh > 0
    # Tiempo consistente con energía/potencia/eficiencia.
    expected_t = stop.energy_added_kwh / (stop.power_kw * cfg.vehicle.charge_efficiency)
    assert stop.time_hours == pytest.approx(expected_t)
