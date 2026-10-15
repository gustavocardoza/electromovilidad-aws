"""CLI end-to-end del EVRP.

Flujo completo para una instancia:

1. Cargar la instancia de ruteo.
2. Cargar (o entrenar) el modelo de consumo desde la telemetría EV.
3. Construir las matrices de distancia y energía.
4. Optimizar la ruta con OR-Tools (secuencia + cargadores, SOC como
   restricción dura).
5. Calcular el plan de recarga parcial realista (post-proceso).
6. Validar la factibilidad de forma independiente y calcular métricas.
7. Exportar la ruta (CSV) y las métricas (JSON) a ``outputs/``.

Uso::

    python -m src.solve --instance evaluation
    python -m src.solve --instance competition --time-limit 60
    python -m src.solve --instance evaluation --no-recharge
    python -m src.solve --instance all --vehicles 2 --distance haversine

Por defecto las distancias son por red vial OSM (``--distance road``); la
matriz se calcula UNA vez y la usan el solver, la recarga y el validador.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import config, data_loading
from .consumption_model import ConsumptionModel
from .distance import build_distance_matrix, energy_matrix_kwh
from .evrp_model import solve_evrp, VehicleRoute
from .recharge import plan_recharge, RechargePlan
from .validate import validate_route, aggregate_fleet_metrics


def _load_or_train_model(train: bool) -> ConsumptionModel:
    params_path = config.MODELS_DIR / "consumption_params.json"
    if not train and params_path.exists():
        return ConsumptionModel.load(params_path)
    from .consumption_model import fit

    model = fit()
    model.params.to_json(params_path)
    return model


def _route_rows(
    vehicle_id: int,
    route_node_ids: list[str],
    route_global_indices: list[int],
    instance: data_loading.RoutingInstance,
    energy_matrix,
    plan: RechargePlan,
    dist_matrix,
) -> list[dict]:
    """Filas CSV de una ruta: orden, nodo, SOC, energía y acciones de recarga."""
    node_by_id = {n.node_id: n for n in instance.all_nodes}
    added_by_node: dict[str, float] = {}
    time_by_node: dict[str, float] = {}
    power_by_node: dict[str, float] = {}
    for stop in plan.stops:
        added_by_node[stop.node_id] = added_by_node.get(stop.node_id, 0.0) + stop.energy_added_kwh
        time_by_node[stop.node_id] = time_by_node.get(stop.node_id, 0.0) + stop.time_hours
        power_by_node[stop.node_id] = stop.power_kw

    soc = plan.soc_profile_kwh
    cum_drive = 0.0
    rows = []
    for i, nid in enumerate(route_node_ids):
        node = node_by_id[nid]
        arc_energy = 0.0
        arc_km = 0.0
        if i < len(route_node_ids) - 1:
            gi = route_global_indices[i]
            gj = route_global_indices[i + 1]
            arc_energy = float(energy_matrix[gi][gj])
            arc_km = float(dist_matrix[gi][gj])
        rows.append(
            {
                "vehicle": vehicle_id,
                "order": i,
                "node_id": nid,
                "node_type": node.node_type,
                "latitude": node.latitude,
                "longitude": node.longitude,
                "soc_arrival_kwh": round(soc[i], 4) if i < len(soc) else "",
                "arc_energy_kwh": round(arc_energy, 4),
                "arc_km": round(arc_km, 4),
                "cumulative_drive_kwh": round(cum_drive, 4),
                "recharged_kwh": round(added_by_node.get(nid, 0.0), 4),
                "recharge_power_kw": power_by_node.get(nid, ""),
                "recharge_time_h": round(time_by_node.get(nid, 0.0), 4) if nid in time_by_node else "",
            }
        )
        cum_drive += arc_energy
    return rows


def _route_km(route_global_indices: list[int], dist_matrix) -> float:
    """Distancia total (km) de una ruta según la matriz que usó el solver."""
    return sum(
        float(dist_matrix[a][b])
        for a, b in zip(route_global_indices[:-1], route_global_indices[1:])
    )


def _write_fleet_csv(path: Path, all_rows: list[dict]) -> None:
    """Escribe el CSV de la flota (una fila por nodo visitado, con 'vehicle')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


def solve_instance(
    instance: str,
    cfg: config.Config | None = None,
    use_rechargers: bool = True,
    train_model: bool = False,
    use_model_capacity: bool = True,
) -> dict:
    """Resuelve una instancia end-to-end y escribe las salidas. Devuelve métricas.

    Si ``use_model_capacity`` es True y el modelo trae una capacidad estimada,
    se usa esa capacidad de batería; en caso contrario se respeta la de ``cfg``
    (p. ej. cuando el usuario la fija por CLI con ``--battery``).
    """
    cfg = cfg or config.Config()
    inst = data_loading.load_instance(instance)
    model = _load_or_train_model(train_model)

    # Capacidad de batería: la estimada por el modelo, salvo override explícito.
    if use_model_capacity and model.battery_capacity_kwh and model.battery_capacity_kwh > 0:
        cfg.vehicle.battery_capacity_kwh = model.battery_capacity_kwh

    # Matriz única (km): la misma para solver, recarga y validador.
    dist = build_distance_matrix(inst, cfg)
    energy = energy_matrix_kwh(inst, model, dist)

    result = solve_evrp(inst, model, cfg, use_rechargers=use_rechargers, dist_km=dist)
    if not result.routes or not result.feasible and result.status == "NO_SOLUTION":
        metrics = {
            "instance": instance,
            "feasible": False,
            "fleet_size": cfg.solver.fleet_size,
            "distance_mode": cfg.distance_mode,
            "reason": f"OR-Tools: {result.status}",
        }
        _write_metrics(config.OUTPUTS_DIR / f"metrics_{instance}.json", metrics)
        return metrics

    # Post-proceso de recarga y validación por vehículo.
    per_vehicle_metrics = []
    per_vehicle_n_visits = []
    all_rows: list[dict] = []
    routes_out = []
    for r in result.routes:
        plan = plan_recharge(r.node_ids, r.node_indices, inst, energy, cfg)
        rm = validate_route(
            inst, r.node_ids, r.node_indices, energy, plan, cfg,
            require_full_coverage=False,  # cobertura se evalúa a nivel de flota
        )
        per_vehicle_metrics.append(rm)
        per_vehicle_n_visits.append(r.n_visits)
        all_rows.extend(
            _route_rows(r.vehicle_id, r.node_ids, r.node_indices, inst, energy, plan, dist)
        )
        routes_out.append(
            {
                "vehicle": r.vehicle_id,
                "route": r.node_ids,
                "chargers_used": r.chargers_used,
                "n_visits": r.n_visits,
                "drive_energy_kwh": r.drive_energy_kwh,
                "drive_km": round(_route_km(r.node_indices, dist), 4),
                "drive_time_hours": round(r.drive_time_hours, 4),
            }
        )

    fleet = aggregate_fleet_metrics(instance, per_vehicle_metrics, per_vehicle_n_visits, cfg)
    for pv, rv in zip(fleet.per_vehicle, routes_out):
        pv["drive_km"] = rv["drive_km"]
        pv["drive_time_hours"] = rv["drive_time_hours"]

    # Exportar CSV de rutas de la flota.
    route_csv = config.OUTPUTS_DIR / f"route_{instance}.csv"
    _write_fleet_csv(route_csv, all_rows)

    metrics = fleet.to_dict()
    metrics["distance_mode"] = cfg.distance_mode
    metrics["total_drive_km"] = round(
        sum(_route_km(r.node_indices, dist) for r in result.routes), 4
    )
    metrics["makespan_weight"] = cfg.solver.makespan_weight
    metrics["avg_speed_kmh"] = cfg.solver.avg_speed_kmh
    # Makespan de conducción: tiempo del vehículo que más tarda (h) y balance.
    metrics["makespan_drive_hours"] = round(result.makespan_drive_hours, 4)
    drive_times = [r.drive_time_hours for r in result.routes]
    metrics["total_drive_time_hours"] = round(sum(drive_times), 4)
    metrics["min_drive_time_hours"] = round(min(drive_times), 4) if drive_times else 0.0
    metrics["routes"] = routes_out
    _write_metrics(config.OUTPUTS_DIR / f"metrics_{instance}.json", metrics)

    return metrics


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def _print_summary(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Instancia: {metrics.get('instance')}  |  Flota: {metrics.get('fleet_size')} vehículo(s)"
          f"  |  Distancias: {metrics.get('distance_mode', '?')}")
    if not metrics.get("feasible"):
        print(f"  NO FACTIBLE: {metrics.get('reason', '')}")
        return
    print(f"  FACTIBLE: SOC mínimo (flota) = {metrics['min_soc_frac'] * 100:.1f}% "
          f"(límite {metrics['min_soc_limit_frac'] * 100:.0f}%)")
    print(f"  Vehículos usados:        {metrics['n_vehicles_used']}/{metrics['fleet_size']}")
    print(f"  Visitas cubiertas:       {metrics['n_visits_covered']}/{metrics['n_visits_required']}")
    print(f"  Energía total conducción:{metrics['total_drive_energy_kwh']:.2f} kWh")
    if "total_drive_km" in metrics:
        print(f"  Distancia total:         {metrics['total_drive_km']:.2f} km")
    print(f"  Recargas (total):        {metrics['total_recharges']}")
    print(f"  Energía repuesta:        {metrics['total_recharge_energy_kwh']:.2f} kWh")
    print(f"  Tiempo de recarga:       {metrics['total_recharge_time_hours'] * 60:.1f} min")
    print(f"  Capacidad batería:       {metrics['battery_capacity_kwh']:.1f} kWh")
    if metrics.get("fleet_size", 1) > 1:
        print(f"  Equilibrio (makespan_weight={metrics.get('makespan_weight', 0)}):")
        print(f"    Makespan conducción:   {metrics.get('makespan_drive_hours', 0) * 60:.1f} min "
              f"(el que más tarda; min {metrics.get('min_drive_time_hours', 0) * 60:.1f} min)")
    print("  Por vehículo:")
    for rv in metrics.get("routes", []):
        chg = f", cargadores: {rv['chargers_used']}" if rv["chargers_used"] else ""
        km = f", {rv['drive_km']:.2f} km" if rv.get("drive_km") is not None else ""
        t = f", {rv['drive_time_hours'] * 60:.0f} min" if rv.get("drive_time_hours") is not None else ""
        print(
            f"    - V{rv['vehicle']}: {rv['n_visits']} visitas, "
            f"{rv['drive_energy_kwh']:.2f} kWh{km}{t}{chg}"
        )
    print("=" * 60)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Resuelve el EVRP end-to-end.")
    parser.add_argument(
        "--instance",
        default="evaluation",
        choices=sorted(config.INSTANCE_FILES) + ["all"],
        help="Instancia a resolver ('all' resuelve ambas).",
    )
    parser.add_argument("--time-limit", type=int, default=None, help="Límite de tiempo (s).")
    parser.add_argument("--vehicles", type=int, default=None, help="Número de vehículos de la flota.")
    parser.add_argument(
        "--makespan-weight", type=float, default=None,
        help="Peso del equilibrio de carga (makespan). 0 = solo energía; "
             ">0 equilibra el tiempo del vehículo que más tarda. Típico 0.5-5.",
    )
    parser.add_argument("--battery", type=float, default=None, help="Capacidad batería (kWh).")
    parser.add_argument("--min-soc", type=float, default=None, help="SOC mínimo (fracción 0-1).")
    parser.add_argument("--no-recharge", action="store_true", help="Resolver sin cargadores.")
    parser.add_argument("--train", action="store_true", help="Reentrenar el modelo de consumo.")
    parser.add_argument(
        "--distance",
        default=None,
        choices=config.DISTANCE_MODES,
        help="Distancias: 'road' (red vial OSM, por defecto) o 'haversine' (línea recta).",
    )
    args = parser.parse_args()

    cfg = config.Config()
    if args.distance is not None:
        cfg.distance_mode = args.distance
    if args.time_limit is not None:
        cfg.solver.time_limit_s = args.time_limit
    if args.vehicles is not None:
        cfg.solver.fleet_size = args.vehicles
    if args.makespan_weight is not None:
        cfg.solver.makespan_weight = args.makespan_weight
    user_set_battery = args.battery is not None
    if user_set_battery:
        cfg.vehicle.battery_capacity_kwh = args.battery
    if args.min_soc is not None:
        cfg.vehicle.min_soc_frac = args.min_soc

    instances = (
        sorted(config.INSTANCE_FILES) if args.instance == "all" else [args.instance]
    )
    for inst_name in instances:
        metrics = solve_instance(
            inst_name,
            cfg=cfg,
            use_rechargers=not args.no_recharge,
            train_model=args.train,
            use_model_capacity=not user_set_battery,
        )
        _print_summary(metrics)
        print(f"  Salidas: outputs/route_{inst_name}.csv, outputs/metrics_{inst_name}.json")


if __name__ == "__main__":
    _main()
