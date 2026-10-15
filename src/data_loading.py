"""Carga de datos: nodos de ruteo y telemetría de vehículos eléctricos.

Expone estructuras tipadas para la instancia de ruteo (base, visitas,
cargadores) y utilidades para cargar la telemetría de EVs usada por el modelo
de consumo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pandas as pd

from . import config


# Columnas esperadas en los archivos de nodos de ruteo.
NODE_COLUMNS = [
    "node_id",
    "node_type",
    "latitude",
    "longitude",
    "address",
    "network",
    "total_ports",
    "dc_fast_ports",
    "level2_ports",
    "connector_types",
    "max_power_kw",
]


@dataclass(frozen=True)
class Node:
    """Un nodo de la instancia de ruteo."""

    node_id: str
    node_type: str  # 'base' | 'visit' | 'charger'
    latitude: float
    longitude: float
    # Solo relevantes para cargadores:
    max_power_kw: float | None = None
    connector_types: tuple[str, ...] = ()
    network: str | None = None
    address: str | None = None

    @property
    def is_base(self) -> bool:
        return self.node_type == "base"

    @property
    def is_visit(self) -> bool:
        return self.node_type == "visit"

    @property
    def is_charger(self) -> bool:
        return self.node_type == "charger"


@dataclass
class RoutingInstance:
    """Instancia completa de ruteo: base + visitas + cargadores."""

    name: str
    base: Node
    visits: list[Node] = field(default_factory=list)
    chargers: list[Node] = field(default_factory=list)

    @property
    def all_nodes(self) -> list[Node]:
        """Nodos en orden canónico: base, visitas, cargadores."""
        return [self.base, *self.visits, *self.chargers]

    def summary(self) -> str:
        powers = [c.max_power_kw for c in self.chargers if c.max_power_kw is not None]
        pmin = min(powers) if powers else float("nan")
        pmax = max(powers) if powers else float("nan")
        return (
            f"Instancia '{self.name}': "
            f"{1} base, {len(self.visits)} visitas, {len(self.chargers)} cargadores. "
            f"Potencia cargadores: {pmin:.1f}-{pmax:.1f} kW."
        )


def _parse_connectors(raw: object) -> tuple[str, ...]:
    """Convierte 'CCS; NACS' -> ('CCS', 'NACS'). Tolera vacíos/NaN."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(tok.strip() for tok in text.split(";") if tok.strip())


def _node_from_row(row: pd.Series) -> Node:
    power = row.get("max_power_kw")
    power_val = None if pd.isna(power) else float(power)
    return Node(
        node_id=str(row["node_id"]).strip(),
        node_type=str(row["node_type"]).strip().lower(),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        max_power_kw=power_val,
        connector_types=_parse_connectors(row.get("connector_types")),
        network=(None if pd.isna(row.get("network")) else str(row.get("network"))),
        address=(None if pd.isna(row.get("address")) else str(row.get("address"))),
    )


def load_instance(instance: str) -> RoutingInstance:
    """Carga una instancia de ruteo por nombre ('evaluation' | 'competition').

    Separa los nodos por ``node_type`` y valida que exista exactamente una base.
    Los cargadores faltantes en la numeración (p. ej. C21, C57) no son un
    problema: simplemente no están en el archivo y no se cargan.
    """
    if instance not in config.INSTANCE_FILES:
        raise ValueError(
            f"Instancia desconocida: {instance!r}. "
            f"Opciones: {sorted(config.INSTANCE_FILES)}"
        )
    path = config.INSTANCE_FILES[instance]
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de nodos: {path}")

    df = pd.read_csv(path)
    missing = [c for c in NODE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path.name}: {missing}")

    # Normaliza max_power_kw a numérico (puede venir con celdas vacías).
    df["max_power_kw"] = pd.to_numeric(df["max_power_kw"], errors="coerce")

    nodes = [_node_from_row(row) for _, row in df.iterrows()]

    bases = [n for n in nodes if n.is_base]
    visits = [n for n in nodes if n.is_visit]
    chargers = [n for n in nodes if n.is_charger]

    if len(bases) != 1:
        raise ValueError(
            f"Se esperaba exactamente 1 nodo base en {path.name}, "
            f"se encontraron {len(bases)}."
        )

    # Orden estable de visitas y cargadores por id numérico (V1, V2, ..., C1...).
    def _sort_key(n: Node) -> tuple[str, int]:
        prefix = n.node_id[0]
        num = "".join(ch for ch in n.node_id if ch.isdigit())
        return (prefix, int(num) if num else 0)

    visits.sort(key=_sort_key)
    chargers.sort(key=_sort_key)

    return RoutingInstance(name=instance, base=bases[0], visits=visits, chargers=chargers)


def load_ev_telemetry(nrows: int | None = None) -> pd.DataFrame:
    """Carga la telemetría de EVs usada para calibrar el modelo de consumo.

    Parameters
    ----------
    nrows:
        Si se indica, limita el número de filas leídas (útil para tests rápidos).
    """
    path = config.EV_TELEMETRY_FILE
    if not path.exists():
        raise FileNotFoundError(f"No existe la telemetría EV: {path}")
    df = pd.read_csv(path, nrows=nrows)
    return df


def load_vehicle_types() -> pd.DataFrame:
    """Carga el catálogo de tipos de vehículo (incluye peso y EngineType)."""
    path = config.VEHICLE_TYPES_FILE
    if not path.exists():
        raise FileNotFoundError(f"No existe vehicle_types: {path}")
    return pd.read_csv(path)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Resumen de una instancia de ruteo.")
    parser.add_argument(
        "--instance",
        default="evaluation",
        choices=sorted(config.INSTANCE_FILES),
        help="Instancia a cargar.",
    )
    args = parser.parse_args()
    inst = load_instance(args.instance)
    print(inst.summary())
    print(f"  Base: {inst.base.node_id} @ ({inst.base.latitude}, {inst.base.longitude})")
    print(f"  Visitas: {[n.node_id for n in inst.visits]}")
    print(f"  Cargadores: {[c.node_id for c in inst.chargers]}")


if __name__ == "__main__":
    _main()
