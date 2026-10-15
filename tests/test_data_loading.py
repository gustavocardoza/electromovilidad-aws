"""Tests de carga de datos (Task 1)."""

import math

import pytest

from src import data_loading


def test_load_evaluation_instance_counts():
    inst = data_loading.load_instance("evaluation")
    # 1 base, 20 visitas.
    assert inst.base.is_base
    assert len(inst.visits) == 20
    assert all(v.is_visit for v in inst.visits)
    # Hay cargadores y todos tienen potencia numérica positiva.
    assert len(inst.chargers) > 0
    for c in inst.chargers:
        assert c.is_charger
        assert c.max_power_kw is not None
        assert c.max_power_kw > 0
        assert not math.isnan(c.max_power_kw)


def test_load_competition_instance_counts():
    inst = data_loading.load_instance("competition")
    assert inst.base.is_base
    assert len(inst.visits) == 20
    assert len(inst.chargers) > 0


def test_visits_sorted_v1_to_v20():
    inst = data_loading.load_instance("evaluation")
    ids = [v.node_id for v in inst.visits]
    assert ids[0] == "V1"
    assert ids[-1] == "V20"


def test_connectors_parsed():
    inst = data_loading.load_instance("evaluation")
    # Al menos un cargador declara CCS.
    assert any("CCS" in c.connector_types for c in inst.chargers)
    # Un cargador J1772 típico.
    assert any("J1772" in c.connector_types for c in inst.chargers)


def test_all_nodes_order():
    inst = data_loading.load_instance("evaluation")
    nodes = inst.all_nodes
    assert nodes[0].is_base
    assert len(nodes) == 1 + len(inst.visits) + len(inst.chargers)


def test_unknown_instance_raises():
    with pytest.raises(ValueError):
        data_loading.load_instance("nope")
