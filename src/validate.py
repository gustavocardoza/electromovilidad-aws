"""Validador independiente de factibilidad y cálculo de métricas.

Recorre una ruta ya resuelta (con su plan de recarga) de forma totalmente
independiente del solver OR-Tools, resimulando el SOC paso a paso. Es la
**fuente de verdad** de factibilidad: detecta inconsistencias de escalado o
redondeo entre el modelo entero de OR-Tools y la energía real.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

from . import config, data_loading
from .recharge import RechargePlan


@dataclass
class RouteMetrics:
    """Métricas de una ruta resuelta."""

    instance: str
    feasible: bool
    n_visits_covered: int
    n_visits_required: int
    all_visits_covered: bool
    drive_energy_kwh: float
    n_recharges: int
    recharge_energy_kwh: float
    recharge_time_hours: float
    min_soc_kwh: float
    min_soc_frac: float
    battery_capacity_kwh: float
    min_soc_limit_frac: float
    plan_feasible: bool = True      # factibilidad del SOC de ESTA ruta
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_route(
    instance: data_loading.RoutingInstance,
    route_node_ids: list[str],
    route_global_indices: list[int],
    energy_matrix_kwh,
    plan: RechargePlan,
    cfg: config.Config | None = None,
    require_full_coverage: bool = True,
) -> RouteMetrics:
    """Resimula el SOC a lo largo de la ruta y calcula métricas.

    La simulación aplica, en cada cargador visitado, la energía repuesta que
    indica ``plan`` (no recarga "ideal"), y resta la energía de conducción de
    cada arco. Verifica que el SOC nunca baje del mínimo.

    ``require_full_coverage`` controla si la factibilidad exige que ESTA ruta
    cubra todas las visitas de la instancia. Para una flota se pone en False (la
    cobertura se evalúa a nivel de flota); el campo ``plan_feasible`` siempre
    refleja únicamente la factibilidad del SOC de la ruta.
    """
    cfg = cfg or config.Config()
    veh = cfg.vehicle
    cap = veh.battery_capacity_kwh
    min_soc = veh.min_soc_kwh

    charger_ids = {c.node_id for c in instance.chargers}
    # Mapa node_id -> energía repuesta (sumada por si se visita 2 veces).
    added_by_node: dict[str, float] = {}
    for stop in plan.stops:
        added_by_node[stop.node_id] = added_by_node.get(stop.node_id, 0.0) + stop.energy_added_kwh

    n = len(route_node_ids)
    soc = veh.initial_soc_kwh
    min_soc_seen = soc
    soc_feasible = True
    reason = ""

    # Consumo del arco: energía de la matriz entre nodos consecutivos.
    for i in range(n):
        node_id = route_node_ids[i]
        # Recarga en cargador (según plan), acotada por la capacidad.
        if node_id in charger_ids and node_id in added_by_node:
            soc = min(cap, soc + added_by_node[node_id])
        # Consumo del arco hacia el siguiente nodo.
        if i < n - 1:
            gi = route_global_indices[i]
            gj = route_global_indices[i + 1]
            soc -= float(energy_matrix_kwh[gi][gj])
            min_soc_seen = min(min_soc_seen, soc)
            if soc < min_soc - 1e-6:
                soc_feasible = False
                reason = (
                    f"SOC {soc:.3f} kWh < mínimo {min_soc:.3f} kWh tras arco "
                    f"{route_node_ids[i]}->{route_node_ids[i + 1]}."
                )
                break

    # SOC factible = sin violar el mínimo Y plan de recarga factible.
    plan_feasible = soc_feasible and plan.feasible

    # Cobertura de visitas (de ESTA ruta respecto a toda la instancia).
    required = {v.node_id for v in instance.visits}
    covered = {nid for nid in route_node_ids if nid in required}
    all_covered = covered == required

    # Factibilidad reportada: depende de si se exige cobertura completa.
    feasible = plan_feasible
    if require_full_coverage and not all_covered:
        feasible = False
        if not reason:
            reason = f"Faltan visitas: {sorted(required - covered)}"

    return RouteMetrics(
        instance=instance.name,
        feasible=feasible,
        n_visits_covered=len(covered),
        n_visits_required=len(required),
        all_visits_covered=all_covered,
        drive_energy_kwh=_route_drive_energy(route_global_indices, energy_matrix_kwh),
        n_recharges=len(plan.stops),
        recharge_energy_kwh=plan.total_recharge_energy_kwh,
        recharge_time_hours=plan.total_recharge_time_hours,
        min_soc_kwh=min_soc_seen,
        min_soc_frac=min_soc_seen / cap if cap > 0 else 0.0,
        battery_capacity_kwh=cap,
        min_soc_limit_frac=veh.min_soc_frac,
        plan_feasible=plan_feasible,
        reason=reason or plan.reason,
    )


def _route_drive_energy(route_global_indices: list[int], energy_matrix_kwh) -> float:
    total = 0.0
    for i in range(len(route_global_indices) - 1):
        total += float(
            energy_matrix_kwh[route_global_indices[i]][route_global_indices[i + 1]]
        )
    return total


@dataclass
class FleetMetrics:
    """Métricas agregadas de una flota + métricas por vehículo."""

    instance: str
    feasible: bool
    fleet_size: int
    n_vehicles_used: int
    n_visits_covered: int
    n_visits_required: int
    all_visits_covered: bool
    total_drive_energy_kwh: float
    total_recharges: int
    total_recharge_energy_kwh: float
    total_recharge_time_hours: float
    min_soc_frac: float            # SOC mínimo sobre toda la flota
    battery_capacity_kwh: float
    min_soc_limit_frac: float
    per_vehicle: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_fleet_metrics(
    instance_name: str,
    per_vehicle_metrics: list[RouteMetrics],
    per_vehicle_n_visits: list[int],
    cfg: "config.Config",
) -> FleetMetrics:
    """Agrega las métricas de cada ruta en métricas de flota.

    La cobertura de visitas se evalúa a nivel de flota (la suma de las rutas
    debe cubrir todas las visitas), no por ruta individual.
    """
    veh = cfg.vehicle
    required = per_vehicle_metrics[0].n_visits_required if per_vehicle_metrics else 0
    total_covered = sum(m.n_visits_covered for m in per_vehicle_metrics)
    all_covered = total_covered == required

    # Factibilidad por SOC: ninguna ruta puede violar el mínimo. Se evalúa con el
    # motivo de cada ruta (que ya ignora la cobertura parcial a nivel de ruta).
    soc_ok = all(
        m.min_soc_kwh >= veh.min_soc_kwh - 1e-6 and m.plan_feasible
        for m in per_vehicle_metrics
    )
    feasible = all_covered and soc_ok

    min_soc_frac = (
        min(m.min_soc_frac for m in per_vehicle_metrics)
        if per_vehicle_metrics
        else 0.0
    )
    n_used = sum(1 for nv in per_vehicle_n_visits if nv > 0)

    reason = ""
    if not all_covered:
        reason = f"Cobertura de flota incompleta: {total_covered}/{required} visitas."
    elif not soc_ok:
        bad = next((m for m in per_vehicle_metrics if not m.plan_feasible
                    or m.min_soc_kwh < veh.min_soc_kwh - 1e-6), None)
        if bad is not None:
            reason = bad.reason

    per_vehicle = []
    for m, nv in zip(per_vehicle_metrics, per_vehicle_n_visits):
        d = m.to_dict()
        d["n_visits"] = nv
        per_vehicle.append(d)

    return FleetMetrics(
        instance=instance_name,
        feasible=feasible,
        fleet_size=cfg.solver.fleet_size,
        n_vehicles_used=n_used,
        n_visits_covered=total_covered,
        n_visits_required=required,
        all_visits_covered=all_covered,
        total_drive_energy_kwh=sum(m.drive_energy_kwh for m in per_vehicle_metrics),
        total_recharges=sum(m.n_recharges for m in per_vehicle_metrics),
        total_recharge_energy_kwh=sum(m.recharge_energy_kwh for m in per_vehicle_metrics),
        total_recharge_time_hours=sum(m.recharge_time_hours for m in per_vehicle_metrics),
        min_soc_frac=min_soc_frac,
        battery_capacity_kwh=veh.battery_capacity_kwh,
        min_soc_limit_frac=veh.min_soc_frac,
        per_vehicle=per_vehicle,
        reason=reason,
    )
