"""Tests del modelo de consumo (Task 2)."""

import numpy as np

from src import consumption_model as cm


def _small_model():
    # Entrena con una muestra reducida para velocidad del test.
    from src import data_loading

    df = data_loading.load_ev_telemetry(nrows=40000)
    return cm.fit(df)


def test_kwh_per_km_positive_and_reasonable():
    model = _small_model()
    # Debe ser positivo y en un rango amplio pero físico para EV urbano
    # (incluye consumo de accesorios/ralentí, por eso puede superar 0.30).
    assert 0.1 < model.kwh_per_km < 0.6


def test_energy_monotonic_and_nonnegative():
    model = cm.ConsumptionModel.default()
    # Monótona creciente.
    assert model.energy(1.0) < model.energy(2.0) < model.energy(10.0)
    # No negativa.
    assert model.energy(0.0) == 0.0
    assert model.energy(5.0) > 0.0


def test_energy_vectorized():
    model = cm.ConsumptionModel.default()
    out = model.energy(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)
    assert np.all(out >= 0)
    assert out[2] > out[1] > out[0]


def test_capacity_estimate_positive():
    model = _small_model()
    assert model.battery_capacity_kwh > 0


def test_default_model():
    model = cm.ConsumptionModel.default()
    assert model.kwh_per_km > 0
    assert model.battery_capacity_kwh > 0
