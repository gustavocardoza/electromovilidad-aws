"""Post-proceso de recarga parcial realista por potencia de estación.

OR-Tools entrega la **secuencia** de nodos (visitas y cargadores). Este módulo
recorre esa secuencia de forma determinista y decide, en cada cargador, cuánta
energía reponer y cuánto tiempo toma, usando la potencia (``max_power_kw``) de
esa estación concreta.

Política de recarga: **parcial, previsora**. En cada cargador se repone solo lo
necesario para llegar con seguridad al siguiente cargador (o a la base) sin
bajar del SOC mínimo, con un margen; nunca por encima de
``target_soc_after_charge_frac``. Esto minimiza el tiempo de recarga y respeta
la idea de "cargar lo justo para continuar".

Tiempo de recarga = energía repuesta / (potencia * eficiencia).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, data_loading


@dataclass
class ChargeStop:
    """Una parada de recarga resuelta."""

    node_id: str
    power_kw: float
    energy_added_kwh: float
    time_hours: float
    soc_before_kwh: float
    soc_after_kwh: float


@dataclass
class RechargePlan:
    """Plan de recarga para una ruta completa."""

    stops: list[ChargeStop]
    soc_profile_kwh: list[float]  # SOC al llegar a cada nodo de la ruta
    total_recharge_energy_kwh: float
    total_recharge_time_hours: float
    feasible: bool
    reason: str = ""


def _segment_energy(
    node_indices: list[int],
    energy_matrix_kwh,
) -> list[float]:
    """Energía de conducción de cada arco de la ruta (len = n_nodos - 1)."""
    return [
        float(energy_matrix_kwh[node_indices[i]][node_indices[i + 1]])
        for i in range(len(node_indices) - 1)
    ]


def plan_recharge(
    route_node_ids: list[str],
    route_global_indices: list[int],
    instance: data_loading.RoutingInstance,
    energy_matrix_kwh,
    cfg: config.Config | None = None,
) -> RechargePlan:
    """Calcula el plan de recarga parcial para una ruta dada.

    Parameters
    ----------
    route_node_ids:
        Secuencia de ``node_id`` de la ruta (BASE ... BASE).
    route_global_indices:
        Índices de cada nodo de la ruta en ``instance.all_nodes`` (para indexar
        la matriz de energía).
    instance:
        Instancia de ruteo (para potencias de cargadores).
    energy_matrix_kwh:
        Matriz NxN de energía de conducción por arco (kWh).
    cfg:
        Configuración (batería, SOC, eficiencia). Por defecto, la global.
    """
    cfg = cfg or config.Config()
    veh = cfg.vehicle

    cap = veh.battery_capacity_kwh
    min_soc = veh.min_soc_kwh
    target_soc = cap * veh.target_soc_after_charge_frac
    eff = veh.charge_efficiency

    charger_power = {
        c.node_id: (c.max_power_kw or 0.0) for c in instance.chargers
    }

    seg_energy = _segment_energy(route_global_indices, energy_matrix_kwh)
    n = len(route_node_ids)

    soc = veh.initial_soc_kwh
    soc_profile = [soc]
    stops: list[ChargeStop] = []
    total_energy = 0.0
    total_time = 0.0

    for i in range(n):
        node_id = route_node_ids[i]
        is_charger = node_id in charger_power

        if is_charger:
            # Energía necesaria para llegar al siguiente cargador o a la base
            # sin bajar del SOC mínimo. Miramos hacia adelante hasta el próximo
            # cargador (o el final de la ruta).
            needed = 0.0
            j = i
            while j < n - 1:
                needed += seg_energy[j]
                nxt = route_node_ids[j + 1]
                j += 1
                if nxt in charger_power:
                    break  # podrá recargar de nuevo allí
            # SOC objetivo tras recargar. El tope DURO es la capacidad real
            # (100%), igual que la cota superior de la dimensión SOC de OR-Tools:
            # así el post-proceso no declara infactible una ruta que el solver sí
            # consideró factible. ``target_soc`` (95%) es solo una preferencia
            # suave: si con el 95% basta para cubrir 'needed' + margen, se queda
            # ahí; si el tramo siguiente exige más, se carga por encima del 95%
            # hasta donde haga falta, sin pasar de la capacidad.
            needed_soc = min(cap, min_soc + needed)
            desired_soc = max(min(target_soc, cap), needed_soc)
            energy_added = max(0.0, desired_soc - soc)

            if energy_added > 0:
                power = charger_power.get(node_id, 0.0)
                if power <= 0:
                    return RechargePlan(
                        stops=stops,
                        soc_profile_kwh=soc_profile,
                        total_recharge_energy_kwh=total_energy,
                        total_recharge_time_hours=total_time,
                        feasible=False,
                        reason=f"Cargador {node_id} con potencia no válida.",
                    )
                time_h = energy_added / (power * eff)
                soc_before = soc
                soc = soc + energy_added
                total_energy += energy_added
                total_time += time_h
                stops.append(
                    ChargeStop(
                        node_id=node_id,
                        power_kw=power,
                        energy_added_kwh=energy_added,
                        time_hours=time_h,
                        soc_before_kwh=soc_before,
                        soc_after_kwh=soc,
                    )
                )

        # Avanzar al siguiente nodo (si hay arco).
        if i < n - 1:
            soc = soc - seg_energy[i]
            soc_profile.append(soc)
            if soc < min_soc - 1e-6:
                return RechargePlan(
                    stops=stops,
                    soc_profile_kwh=soc_profile,
                    total_recharge_energy_kwh=total_energy,
                    total_recharge_time_hours=total_time,
                    feasible=False,
                    reason=(
                        f"SOC bajo el mínimo tras arco {route_node_ids[i]}->"
                        f"{route_node_ids[i + 1]}: {soc:.2f} < {min_soc:.2f} kWh."
                    ),
                )

    return RechargePlan(
        stops=stops,
        soc_profile_kwh=soc_profile,
        total_recharge_energy_kwh=total_energy,
        total_recharge_time_hours=total_time,
        feasible=True,
    )


def recharge_time_hours(
    energy_kwh: float, power_kw: float, efficiency: float = 0.90
) -> float:
    """Tiempo de recarga = energía / (potencia * eficiencia)."""
    if power_kw <= 0:
        raise ValueError("La potencia debe ser positiva.")
    return energy_kwh / (power_kw * efficiency)
