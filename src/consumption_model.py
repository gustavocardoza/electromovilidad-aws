"""Modelo de consumo de energía calibrado desde la telemetría EV.

La telemetría ``eVED_EV.csv`` trae ``Energy_Consumption`` por muestra (en kWh,
puede ser negativo por regeneración). Agregamos por viaje (``VehId``, ``Trip``),
calculamos distancia recorrida con Haversine entre muestras consecutivas, y
ajustamos un modelo de consumo por kilómetro.

Estimador principal (robusto): regresión de energía total de viaje sobre
distancia total, **sin intercepto y ponderada por distancia**. Esto equivale a
``kWh/km = sum(energia_i) / sum(distancia_i)`` sobre los viajes aceptados, que es
mucho más estable que promediar los ratios por viaje (dominados por viajes
cortos con mucho ralentí/regeneración).

Modelo enriquecido (opcional): regresión lineal de kWh/km del viaje sobre
velocidad media y pendiente media, útil para análisis; para el EVRP se usa un
único coeficiente kWh/km porque las instancias no traen velocidad/pendiente por
arco.

También estima la capacidad de batería a partir de ``energia_viaje / ΔSOC``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit

from . import config
from . import data_loading


# Límites físicos plausibles para un EV compacto (kWh/km). Se usan para avisar
# si la estimación cae fuera de rango, no para forzarla.
PLAUSIBLE_KWH_PER_KM = (0.12, 0.30)


@dataclass
class ConsumptionParams:
    """Parámetros persistibles del modelo de consumo."""

    kwh_per_km: float
    battery_capacity_kwh: float
    n_trips_used: int
    total_distance_km: float
    total_energy_kwh: float
    # Modelo enriquecido (coeficientes informativos; no usados por el EVRP).
    speed_coef: float | None = None
    gradient_coef: float | None = None
    intercept: float | None = None
    # Diagnóstico.
    kwh_per_km_median_ratio: float | None = None
    within_plausible_range: bool | None = None

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ConsumptionParams":
        return cls(**json.loads(Path(path).read_text()))


class ConsumptionModel:
    """Modelo de consumo entrenado. Predice energía (kWh) a partir de km."""

    def __init__(self, params: ConsumptionParams):
        self.params = params

    @property
    def kwh_per_km(self) -> float:
        return self.params.kwh_per_km

    @property
    def battery_capacity_kwh(self) -> float:
        return self.params.battery_capacity_kwh

    def energy(self, distance_km: float | np.ndarray) -> float | np.ndarray:
        """Energía de conducción (kWh) para una distancia dada (km).

        Monótona y no negativa: ``max(0, kWh/km) * distancia``. Nunca negativa
        porque en el ruteo no modelamos regeneración neta entre nodos.
        """
        d = np.asarray(distance_km, dtype=float)
        e = self.kwh_per_km * d
        e = np.maximum(e, 0.0)
        return float(e) if np.isscalar(distance_km) or d.ndim == 0 else e

    @classmethod
    def default(cls) -> "ConsumptionModel":
        """Modelo con valores por defecto, sin entrenar (fallback)."""
        return cls(
            ConsumptionParams(
                kwh_per_km=0.18,
                battery_capacity_kwh=config.DEFAULT_CONFIG.vehicle.battery_capacity_kwh,
                n_trips_used=0,
                total_distance_km=0.0,
                total_energy_kwh=0.0,
                within_plausible_range=True,
            )
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "ConsumptionModel":
        path = path or (config.MODELS_DIR / "consumption_params.json")
        return cls(ConsumptionParams.from_json(path))


def _trip_distance_km(g: pd.DataFrame) -> float:
    """Distancia Haversine acumulada entre muestras consecutivas de un viaje."""
    if len(g) < 2:
        return 0.0
    coords = g[["Latitude[deg]", "Longitude[deg]"]].to_numpy()
    d = haversine_vector(coords[:-1], coords[1:], unit=Unit.KILOMETERS)
    return float(np.sum(d))


def aggregate_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la telemetría por (VehId, Trip).

    Devuelve un DataFrame con distancia, energía total, ΔSOC y medias de
    velocidad y pendiente por viaje.
    """
    # Filtra muestras con coordenadas o velocidad inválidas.
    df = df.copy()
    df = df[df["Vehicle Speed[km/h]"].fillna(0) >= 0]
    df = df.dropna(subset=["Latitude[deg]", "Longitude[deg]", "Energy_Consumption"])

    records = []
    for (veh, trip), g in df.groupby(["VehId", "Trip"], sort=False):
        g = g.sort_values("Timestamp")
        dist = _trip_distance_km(g)
        soc = g["HV Battery SOC[%]"]
        records.append(
            {
                "VehId": veh,
                "Trip": trip,
                "n_samples": len(g),
                "distance_km": dist,
                "energy_kwh": float(g["Energy_Consumption"].sum()),
                "dsoc_pct": float(soc.iloc[0] - soc.iloc[-1]),
                "speed_mean": float(g["Vehicle Speed[km/h]"].mean()),
                "gradient_mean": float(g["Gradient"].fillna(0).mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def fit(
    df: pd.DataFrame | None = None,
    min_trip_distance_km: float = 0.5,
) -> ConsumptionModel:
    """Entrena el modelo de consumo desde la telemetría.

    Parameters
    ----------
    df:
        Telemetría EV. Si es ``None`` se carga de disco.
    min_trip_distance_km:
        Descarta viajes muy cortos (ruido dominante, mucho ralentí) del ajuste
        del kWh/km.
    """
    if df is None:
        df = data_loading.load_ev_telemetry()

    trips = aggregate_trips(df)

    # Viajes válidos para estimar kWh/km: distancia y energía positivas.
    valid = trips[
        (trips["distance_km"] >= min_trip_distance_km)
        & (trips["energy_kwh"] > 0)
    ].copy()

    total_dist = float(valid["distance_km"].sum())
    total_energy = float(valid["energy_kwh"].sum())
    # Estimador robusto ponderado por distancia (regresión sin intercepto).
    kwh_per_km = total_energy / total_dist if total_dist > 0 else 0.18

    valid["ratio"] = valid["energy_kwh"] / valid["distance_km"]
    median_ratio = float(valid["ratio"].median())

    # Capacidad de batería: energía/ΔSOC sobre viajes con descarga apreciable.
    cap_rows = trips[(trips["dsoc_pct"] > 1.0) & (trips["energy_kwh"] > 0)].copy()
    if len(cap_rows) > 0:
        cap_rows["kwh_per_pct"] = cap_rows["energy_kwh"] / cap_rows["dsoc_pct"]
        capacity = float(cap_rows["kwh_per_pct"].median() * 100.0)
    else:
        capacity = config.DEFAULT_CONFIG.vehicle.battery_capacity_kwh

    # Modelo enriquecido informativo: ratio ~ speed_mean + gradient_mean.
    speed_coef = gradient_coef = intercept = None
    enr = valid[valid["ratio"].between(0.05, 1.0)]
    if len(enr) >= 10:
        from sklearn.linear_model import LinearRegression

        X = enr[["speed_mean", "gradient_mean"]].to_numpy()
        y = enr["ratio"].to_numpy()
        lr = LinearRegression().fit(X, y)
        speed_coef = float(lr.coef_[0])
        gradient_coef = float(lr.coef_[1])
        intercept = float(lr.intercept_)

    within = PLAUSIBLE_KWH_PER_KM[0] <= kwh_per_km <= PLAUSIBLE_KWH_PER_KM[1]

    params = ConsumptionParams(
        kwh_per_km=kwh_per_km,
        battery_capacity_kwh=capacity,
        n_trips_used=int(len(valid)),
        total_distance_km=total_dist,
        total_energy_kwh=total_energy,
        speed_coef=speed_coef,
        gradient_coef=gradient_coef,
        intercept=intercept,
        kwh_per_km_median_ratio=median_ratio,
        within_plausible_range=bool(within),
    )
    return ConsumptionModel(params)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de consumo.")
    parser.add_argument(
        "--out",
        type=Path,
        default=config.MODELS_DIR / "consumption_params.json",
        help="Ruta de salida de los parámetros (JSON).",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Limitar filas (test).")
    args = parser.parse_args()

    df = data_loading.load_ev_telemetry(nrows=args.nrows)
    model = fit(df)
    model.params.to_json(args.out)
    p = model.params
    print("Modelo de consumo entrenado:")
    print(f"  kWh/km (robusto, ponderado):   {p.kwh_per_km:.4f}")
    print(f"  kWh/km (mediana de ratios):    {p.kwh_per_km_median_ratio:.4f}")
    print(f"  Dentro de rango plausible {PLAUSIBLE_KWH_PER_KM}: {p.within_plausible_range}")
    print(f"  Capacidad batería estimada:    {p.battery_capacity_kwh:.1f} kWh")
    print(f"  Viajes usados:                 {p.n_trips_used}")
    print(f"  Distancia total:               {p.total_distance_km:.1f} km")
    print(f"  Energía total:                 {p.total_energy_kwh:.1f} kWh")
    if p.speed_coef is not None:
        print(
            f"  Modelo enriquecido: ratio = {p.intercept:.4f} "
            f"+ {p.speed_coef:.5f}*v + {p.gradient_coef:.3f}*grad"
        )
    print(f"  Guardado en: {args.out}")


if __name__ == "__main__":
    _main()
