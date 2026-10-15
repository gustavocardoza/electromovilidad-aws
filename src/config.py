"""Parámetros y configuración del solver EVRP.

Centraliza todos los supuestos del modelo (batería, SOC, recarga, conector,
optimización) para que sean reproducibles y fáciles de documentar. Los valores
relacionados con la batería pueden refinarse con los parámetros estimados por
``consumption_model`` (ver ``ConsumptionModel.battery_capacity_kwh``), pero aquí
viven los valores por defecto y los supuestos.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# --- Rutas del proyecto -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

# Mapeo de nombre de instancia -> archivo de nodos de ruteo.
INSTANCE_FILES = {
    "evaluation": DATA_DIR / "evaluation_routing_nodes.csv",
    "competition": DATA_DIR / "competition_routing_nodes.csv",
}

EV_TELEMETRY_FILE = DATA_DIR / "eVED_EV.csv"
VEHICLE_TYPES_FILE = DATA_DIR / "vehicle_types.csv"

# Factor de escala para pasar de kWh (float) a enteros que consume OR-Tools.
# 1 kWh -> 1_000_000 "micro-kWh" (= mWh). Da ~6 cifras significativas, más que
# suficiente para distancias urbanas sin desbordar enteros de 64 bits.
ENERGY_SCALE = 1_000_000

# Modos de distancia entre nodos: 'road' = camino más corto por la red vial de
# OpenStreetMap (dirigido, respeta sentidos); 'haversine' = línea recta.
DISTANCE_MODES = ("road", "haversine")


@dataclass
class VehicleConfig:
    """Supuestos del vehículo eléctrico y de la recarga.

    Attributes
    ----------
    battery_capacity_kwh:
        Capacidad útil de la batería. Valor por defecto estándar de un EV
        compacto; se sobreescribe con la estimación de la telemetría cuando
        esta es estable (ver ``consumption_model``).
    initial_soc_frac:
        Estado de carga inicial como fracción de la capacidad (1.0 = 100%).
    min_soc_frac:
        SOC mínimo permitido (margen de seguridad). Restricción dura: el SOC
        nunca puede bajar de este valor en ningún nodo de la ruta.
    charge_efficiency:
        Eficiencia de recarga (energía que entra a la batería / energía
        entregada por el cargador). Afecta el tiempo de recarga.
    target_soc_after_charge_frac:
        SOC objetivo tras una recarga en el post-proceso. La recarga es parcial:
        se repone lo necesario para el siguiente tramo factible + margen, hasta
        un máximo de este valor.
    connector:
        Conector del EV. Un cargador es compatible si su lista de
        ``connector_types`` incluye alguno de estos tokens.
    """

    # Por defecto ~33 kWh, consistente con la capacidad estimada desde la
    # telemetría EV (eVED_EV.csv, energía/ΔSOC). Se puede sobreescribir con el
    # valor del modelo entrenado vía ConsumptionModel.battery_capacity_kwh.
    battery_capacity_kwh: float = 33.0
    initial_soc_frac: float = 1.0
    min_soc_frac: float = 0.10
    charge_efficiency: float = 0.90
    target_soc_after_charge_frac: float = 0.95
    # El EV puede usar cualquiera de estos conectores (DC rápido CCS o AC J1772).
    connector: tuple[str, ...] = ("CCS", "J1772")

    @property
    def usable_capacity_kwh(self) -> float:
        """Energía utilizable entre SOC inicial y SOC mínimo."""
        return self.battery_capacity_kwh * (self.initial_soc_frac - self.min_soc_frac)

    @property
    def min_soc_kwh(self) -> float:
        return self.battery_capacity_kwh * self.min_soc_frac

    @property
    def initial_soc_kwh(self) -> float:
        return self.battery_capacity_kwh * self.initial_soc_frac


@dataclass
class SolverConfig:
    """Parámetros de la optimización con OR-Tools."""

    time_limit_s: int = 30
    # Número de vehículos de la flota (homogénea). Todas las rutas parten y
    # terminan en la base. Con fleet_size > 1 se reparte el conjunto de visitas
    # entre los vehículos minimizando la energía total.
    fleet_size: int = 1
    # Si True, cada vehículo debe realizar al menos una visita (no se permiten
    # rutas vacías base->base).
    require_all_vehicles_used: bool = True
    # Penalización por NO visitar un cargador (nodo opcional). Debe ser > 0 para
    # que el solver prefiera no usar cargadores innecesarios, pero lo bastante
    # grande como para que acepte usarlos cuando son imprescindibles para la
    # factibilidad del SOC. Se expresa en unidades escaladas de energía.
    charger_skip_penalty: int = 0  # ver evrp_model: se calcula dinámicamente
    # Término pequeño por usar un cargador (coste fijo por parada), en kWh.
    charger_use_cost_kwh: float = 0.05
    first_solution_strategy: str = "PATH_CHEAPEST_ARC"
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH"
    seed: int = 42


@dataclass
class Config:
    """Configuración raíz del solver."""

    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    energy_scale: int = ENERGY_SCALE
    # Distancias del solver: 'road' (red vial OSM, por defecto) o 'haversine'
    # (línea recta, alternativa sin red vial). Ver distance.build_distance_matrix.
    distance_mode: str = "road"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = Config()
