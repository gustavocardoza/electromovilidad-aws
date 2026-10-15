"""Traductor web <-> solver EVRP.

Convierte el JSON que envía la web (base, visitas, cargadores, fleet_size) en los
objetos que entiende el solver, ejecuta el optimizador y traduce el resultado de
vuelta al JSON de respuesta esperado por el backend.

No modifica el solver: solo adapta formatos. El flujo es::

    payload (dict)  --instance_from_request-->  RoutingInstance, fleet_size
    RoutingInstance --solve_evrp (existente)-->  FleetResult
    FleetResult     --response_from_result-->    respuesta (dict)

Decisiones (acordadas con el usuario):
* La salida replica el formato de ejemplo: ``summary`` + ``result.routes`` con
  ``vehicle_id`` / ``stops`` (secuencia real de paradas del solver).
* El traductor genera el sobre: ``request_id`` (uuid), ``created_at`` (UTC),
  ``status``, ``persistence``.
* Los parámetros no incluidos en el input (batería, SOC, modo de distancia, peso
  de makespan) usan los valores por defecto de ``config.Config``.
* Se asume que las coordenadas caen en la zona del grafo OSM cacheado (Michigan,
  Ann Arbor). No se descarga red vial nueva por request.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from . import config, data_loading
from .consumption_model import ConsumptionModel
from .distance import energy_matrix_kwh
from .evrp_model import solve_evrp, FleetResult


# --------------------------------------------------------------------------- #
# Entrada: JSON web -> objetos del solver
# --------------------------------------------------------------------------- #

def _coerce_connectors(raw: Any) -> tuple[str, ...]:
    """Normaliza la lista de conectores del JSON a una tupla de strings."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        # Admite "CCS; NACS" o "CCS,NACS" por tolerancia.
        parts = raw.replace(";", ",").split(",")
        return tuple(p.strip() for p in parts if p.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(p).strip() for p in raw if str(p).strip())
    raise ValueError(f"connector_types con formato inesperado: {raw!r}")


def _node_from_json(obj: dict, node_type: str) -> data_loading.Node:
    """Construye un Node del solver desde un objeto JSON {node_id, lat, lon, ...}."""
    try:
        node_id = str(obj["node_id"]).strip()
        lat = float(obj["lat"])
        lon = float(obj["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Nodo mal formado ({node_type}): {obj!r} ({exc})") from exc

    power = obj.get("max_power_kw")
    power_val = float(power) if power is not None else None
    return data_loading.Node(
        node_id=node_id,
        node_type=node_type,
        latitude=lat,
        longitude=lon,
        max_power_kw=power_val,
        connector_types=_coerce_connectors(obj.get("connector_types")),
    )


def instance_from_request(payload: dict) -> tuple[data_loading.RoutingInstance, int]:
    """Traduce el JSON de entrada a (RoutingInstance, fleet_size).

    Valida los campos mínimos. ``fleet_size`` por defecto 1 si no viene.
    """
    if not isinstance(payload, dict):
        raise ValueError("El payload debe ser un objeto JSON (dict).")
    if "base" not in payload:
        raise ValueError("Falta 'base' en el payload.")

    base = _node_from_json(payload["base"], "base")

    visits_raw = payload.get("visits", []) or []
    chargers_raw = payload.get("chargers", []) or []
    if not isinstance(visits_raw, list) or not isinstance(chargers_raw, list):
        raise ValueError("'visits' y 'chargers' deben ser listas.")

    visits = [_node_from_json(v, "visit") for v in visits_raw]
    chargers = [_node_from_json(c, "charger") for c in chargers_raw]

    raw_fleet = payload.get("fleet_size")
    fleet_size = 1 if raw_fleet is None else int(raw_fleet)
    if fleet_size < 1:
        raise ValueError(f"fleet_size debe ser >= 1, se recibió {fleet_size}.")

    instance = data_loading.RoutingInstance(
        name=f"web-{uuid.uuid4().hex[:8]}",
        base=base,
        visits=visits,
        chargers=chargers,
    )
    return instance, fleet_size


# --------------------------------------------------------------------------- #
# Resolución
# --------------------------------------------------------------------------- #

def _build_config(fleet_size: int, overrides: dict | None = None) -> config.Config:
    """Config por defecto con el fleet_size del request (y overrides opcionales)."""
    cfg = config.Config()
    cfg.solver.fleet_size = fleet_size
    if overrides:
        for key, value in overrides.items():
            # Permite ajustar solver.* o vehicle.* o distance_mode desde código.
            if key == "distance_mode":
                cfg.distance_mode = value
            elif hasattr(cfg.solver, key):
                setattr(cfg.solver, key, value)
            elif hasattr(cfg.vehicle, key):
                setattr(cfg.vehicle, key, value)
    return cfg


def _road_matrix_for_instance(instance: data_loading.RoutingInstance):
    """Matriz de distancias por calles SIN caché por nombre (instancia efímera).

    Para requests web la instancia tiene un nombre único por llamada; usar la
    caché ``.npz`` por nombre dejaría archivos basura. Se calcula directamente
    sobre el grafo cacheado en memoria.
    """
    from . import street_routes
    graph = street_routes.get_drive_graph()
    return street_routes.compute_road_distance_matrix_km(instance, graph)


def solve_request(
    payload: dict,
    model: ConsumptionModel | None = None,
    persistence: str = "dynamodb",
    config_overrides: dict | None = None,
) -> dict:
    """Resuelve un request web completo y devuelve el JSON de respuesta.

    Parameters
    ----------
    payload:
        JSON de entrada (base, visits, chargers, fleet_size).
    model:
        Modelo de consumo. Si es ``None`` se carga el entrenado (o el por
        defecto si no existe en disco).
    persistence:
        Valor del campo ``persistence`` en la respuesta (p. ej. 'dynamodb').
    config_overrides:
        Ajustes opcionales de la Config por defecto (solver.* / vehicle.* /
        distance_mode). El input web NO los trae; es un gancho para pruebas.
    """
    instance, fleet_size = instance_from_request(payload)

    if model is None:
        try:
            model = ConsumptionModel.load()
        except FileNotFoundError:
            model = ConsumptionModel.default()

    cfg = _build_config(fleet_size, config_overrides)

    # Matriz de distancias según el modo (road usa el grafo OSM cacheado).
    if cfg.distance_mode == "road":
        dist = _road_matrix_for_instance(instance)
    else:
        from .distance import build_distance_matrix
        dist = build_distance_matrix(instance, cfg)

    result = solve_evrp(instance, model, cfg, use_rechargers=True, dist_km=dist)
    return response_from_result(instance, fleet_size, result, persistence=persistence)


# --------------------------------------------------------------------------- #
# Salida: resultado del solver -> JSON web
# --------------------------------------------------------------------------- #

def _summary(instance: data_loading.RoutingInstance, fleet_size: int) -> dict:
    return {
        "base_id": instance.base.node_id,
        "visit_count": len(instance.visits),
        "charger_count": len(instance.chargers),
        "fleet_size": fleet_size,
        "visit_ids": [v.node_id for v in instance.visits],
        "charger_ids": [c.node_id for c in instance.chargers],
    }


def _routes_payload(result: FleetResult, fleet_size: int) -> list[dict]:
    """Rutas en el formato del ejemplo: vehicle_id + stops (secuencia de nodos).

    ``vehicle_id`` es 'vehicle-N' (1-indexado) para coincidir con el ejemplo.
    """
    routes = []
    for r in result.routes:
        routes.append(
            {
                "vehicle_id": f"vehicle-{r.vehicle_id + 1}",
                "stops": list(r.node_ids),
            }
        )
    return routes


def response_from_result(
    instance: data_loading.RoutingInstance,
    fleet_size: int,
    result: FleetResult,
    persistence: str = "dynamodb",
    request_id: str | None = None,
    created_at: str | None = None,
) -> dict:
    """Traduce el FleetResult al JSON de respuesta esperado por el backend.

    Genera el sobre (request_id, created_at, status, persistence) y arma
    ``result`` con ``summary`` y ``result.routes`` en el formato del ejemplo.
    """
    request_id = request_id or str(uuid.uuid4())
    created_at = created_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    feasible = result.feasible
    inner_status = "completed" if feasible else result.status.lower()
    message = (
        "Optimización completada."
        if feasible
        else f"El optimizador no encontró solución factible (status={result.status})."
    )

    return {
        "request_id": request_id,
        "created_at": created_at,
        "status": "completed",
        "persistence": persistence,
        "result": {
            "status": inner_status,
            "summary": _summary(instance, fleet_size),
            "result": {
                "message": message,
                "routes": _routes_payload(result, fleet_size),
            },
        },
    }
