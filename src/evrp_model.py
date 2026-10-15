"""Modelo EVRP en OR-Tools: flota de N vehículos + restricción dura de SOC.

Enfoque híbrido (confirmado con el usuario):

* OR-Tools decide **cómo repartir las visitas entre los vehículos**, el orden de
  cada ruta y **qué cargadores usar**, garantizando por construcción que el SOC
  de cada vehículo nunca cae bajo el mínimo.
* Un post-proceso determinista (``recharge.py``) calcula la recarga parcial
  realista por potencia de estación, ruta por ruta.

Flota
-----
* ``fleet_size`` vehículos **homogéneos** (misma batería, consumo y conector).
* Todas las rutas parten y terminan en la base.
* Objetivo global: minimizar la **energía de conducción total** (suma de todas
  las rutas), más (opcional) un término de **makespan** que equilibra la carga.
* Si ``require_all_vehicles_used`` es True, cada vehículo debe hacer al menos
  una visita (no se permiten rutas vacías base->base).

Equilibrio de carga (makespan)
------------------------------
Con el objetivo de solo energía, el reparto óptimo es desigual (un vehículo hace
casi todas las visitas, porque repartir añade viajes de ida/vuelta a la base).
Para equilibrar se añade una dimensión de **tiempo de conducción** (distancia /
``avg_speed_kmh``) y se penaliza su *span* global con
``SetGlobalSpanCostCoefficient``: eso minimiza el tiempo del vehículo que más
tarda (el makespan), empujando a que ningún vehículo quede sobrecargado. El peso
``makespan_weight`` gradúa el equilibrio (0 = solo energía).

Modelado de la batería (dimensión de SOC restante, por vehículo)
----------------------------------------------------------------
La relación de OR-Tools entre nodos consecutivos de una dimensión es::

    cumul(j) = cumul(i) + transit(i) + slack(i)

Modelamos la energía como **SOC restante** (energía disponible en la batería).
OR-Tools mantiene un ``cumul`` independiente por vehículo, así que la dimensión
``soc`` representa el SOC de cada vehículo a lo largo de su ruta:

* El ``cumul`` arranca en ``initial_soc`` en la base (inicio de cada vehículo).
* El ``transit`` de un arco (i->j) es **negativo**: ``-energia(i, j)``.
* El ``slack`` solo es positivo en **cargadores** (recarga). En el resto, 0.
* El ``cumul`` está acotado en ``[min_soc, capacity]``. La cota inferior es la
  **restricción dura**: el SOC nunca baja del mínimo.

Distancias
----------
La energía por arco sale de ``build_distance_matrix`` (red vial OSM por
defecto). Esa matriz es **dirigida** (calles de un sentido): energia(i, j) puede
diferir de energia(j, i). OR-Tools la usa tal cual, sin simetrizar.

Cargadores opcionales
---------------------
Cada cargador compatible es un nodo opcional (``AddDisjunction`` con penalización
0). Cualquier vehículo puede usarlo cuando lo necesita.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from . import config, data_loading
from .consumption_model import ConsumptionModel
from .distance import build_distance_matrix, energy_matrix_kwh, energy_matrix_scaled


@dataclass
class VehicleRoute:
    """Ruta de un vehículo de la flota."""

    vehicle_id: int
    node_ids: list[str]            # secuencia de node_id (BASE al inicio y fin)
    node_indices: list[int]        # índices en instance.all_nodes
    drive_energy_kwh: float        # energía de conducción de esta ruta
    chargers_used: list[str]       # cargadores visitados por este vehículo
    soc_at_nodes_kwh: list[float]  # SOC al llegar a cada nodo de la ruta
    n_visits: int                  # nº de visitas atendidas por este vehículo
    drive_time_hours: float = 0.0  # tiempo de conducción (sin recargas)


@dataclass
class FleetResult:
    """Resultado de la optimización para toda la flota."""

    routes: list[VehicleRoute] = field(default_factory=list)
    total_drive_energy_kwh: float = float("inf")
    feasible: bool = False
    status: str = "NO_SOLUTION"
    # Makespan de conducción: tiempo del vehículo que más tarda conduciendo (h).
    makespan_drive_hours: float = 0.0

    @property
    def all_chargers_used(self) -> list[str]:
        out: list[str] = []
        for r in self.routes:
            out.extend(r.chargers_used)
        return out


def _enum_first_solution(name: str) -> int:
    return getattr(routing_enums_pb2.FirstSolutionStrategy, name)


def _enum_metaheuristic(name: str) -> int:
    return getattr(routing_enums_pb2.LocalSearchMetaheuristic, name)


def _compatible_charger(node: data_loading.Node, connectors: tuple[str, ...]) -> bool:
    """Un cargador es compatible si comparte algún conector con el EV."""
    if not node.connector_types:
        return False
    return any(c in node.connector_types for c in connectors)


def solve_evrp(
    instance: data_loading.RoutingInstance,
    model: ConsumptionModel,
    cfg: config.Config | None = None,
    use_rechargers: bool = True,
    dist_km: np.ndarray | None = None,
) -> FleetResult:
    """Resuelve el EVRP (flota de ``cfg.solver.fleet_size`` vehículos).

    Parameters
    ----------
    instance:
        Instancia de ruteo cargada.
    model:
        Modelo de consumo entrenado (define kWh/km).
    cfg:
        Configuración (batería, SOC, solver, flota). Si ``None`` usa la de
        defecto.
    use_rechargers:
        Si ``False``, resuelve sin cargadores (útil para validar el esqueleto o
        cuando la batería alcanza sin recargas).
    dist_km:
        Matriz NxN de distancias (km) en el orden de ``instance.all_nodes``.
        Puede ser asimétrica (red vial dirigida). Si es ``None`` se construye
        con ``build_distance_matrix(instance, cfg)``. Pasarla permite que el
        solver, la recarga y el validador usen exactamente la misma matriz.
    """
    cfg = cfg or config.Config()
    veh = cfg.vehicle
    scale = cfg.energy_scale
    fleet_size = max(1, int(cfg.solver.fleet_size))

    # --- Nodos del modelo: BASE, visitas (obligatorias), cargadores (opc.) ---
    base = instance.base
    visits = instance.visits
    chargers = (
        [c for c in instance.chargers if _compatible_charger(c, veh.connector)]
        if use_rechargers
        else []
    )

    model_nodes: list[data_loading.Node] = [base, *visits, *chargers]
    n = len(model_nodes)

    base_idx = 0
    visit_idxs = set(range(1, 1 + len(visits)))
    charger_idxs = set(range(1 + len(visits), n))

    # --- Submatrices de energía restringidas a los nodos del modelo ----------
    all_nodes = instance.all_nodes
    id_to_global = {node.node_id: i for i, node in enumerate(all_nodes)}
    global_idx = [id_to_global[nd.node_id] for nd in model_nodes]

    full_dist = dist_km if dist_km is not None else build_distance_matrix(instance, cfg)
    expected_shape = (len(all_nodes), len(all_nodes))
    if np.shape(full_dist) != expected_shape:
        raise ValueError(
            f"dist_km tiene forma {np.shape(full_dist)}, se esperaba {expected_shape}."
        )
    full_energy = energy_matrix_kwh(instance, model, full_dist)
    sub_energy = full_energy[np.ix_(global_idx, global_idx)]
    sub_energy_scaled = energy_matrix_scaled(sub_energy, scale)

    # Submatriz de distancias (km) para derivar tiempos de conducción.
    full_dist_arr = np.asarray(full_dist, dtype=float)
    sub_dist_km = full_dist_arr[np.ix_(global_idx, global_idx)]

    # Enteros escalados de SOC.
    capacity_scaled = int(round(veh.initial_soc_kwh * scale))
    min_soc_scaled = int(round(veh.min_soc_kwh * scale))
    initial_soc_scaled = int(round(veh.initial_soc_kwh * scale))

    # --- OR-Tools: manager con fleet_size vehículos, todos en la base --------
    manager = pywrapcp.RoutingIndexManager(n, fleet_size, base_idx)
    routing = pywrapcp.RoutingModel(manager)

    # Objetivo: energía de conducción (positiva), igual para todos los vehículos.
    def cost_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(sub_energy_scaled[i][j])

    cost_cb_idx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_cb_idx)

    # Dimensión de SOC restante: transit NEGATIVO (consumo descarga la batería).
    def soc_transit_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return -int(sub_energy_scaled[i][j])

    soc_cb_idx = routing.RegisterTransitCallback(soc_transit_cb)

    slack_max = capacity_scaled if chargers else 0
    routing.AddDimension(
        soc_cb_idx,
        slack_max,        # máximo "incremento" de SOC por nodo (recarga)
        capacity_scaled,  # cota superior del SOC (batería llena)
        False,            # NO fijar cumul inicial a 0; lo fijamos nosotros
        "soc",
    )
    soc_dim = routing.GetDimensionOrDie("soc")

    # Cota del SOC en todos los nodos internos: [min_soc, capacity]. Slack = 0
    # salvo en cargadores (donde se puede recargar).
    for node in range(n):
        if node == base_idx:
            continue  # el start/end de cada vehículo se trata aparte
        index = manager.NodeToIndex(node)
        soc_dim.CumulVar(index).SetMin(min_soc_scaled)
        soc_dim.CumulVar(index).SetMax(capacity_scaled)
        if node not in charger_idxs:
            soc_dim.SlackVar(index).SetValue(0)

    # Inicio/fin de cada vehículo: SOC inicial lleno, sin recarga en el arranque.
    for v in range(fleet_size):
        start_index = routing.Start(v)
        end_index = routing.End(v)
        soc_dim.CumulVar(start_index).SetValue(initial_soc_scaled)
        soc_dim.SlackVar(start_index).SetValue(0)
        soc_dim.CumulVar(end_index).SetMin(min_soc_scaled)
        soc_dim.CumulVar(end_index).SetMax(capacity_scaled)

    # --- Cargadores opcionales ----------------------------------------------
    for k in charger_idxs:
        index = manager.NodeToIndex(k)
        routing.AddDisjunction([index], 0)  # penalty 0: visita opcional real

    # --- Forzar que todos los vehículos se usen (al menos una visita) --------
    # Se modela con una dimensión que cuenta visitas: el transit al LLEGAR a un
    # nodo de visita suma 1 (0 en base y cargadores). Exigimos que el cumul al
    # final de cada vehículo sea >= 1, es decir, cada vehículo atiende al menos
    # una visita. (No basta con prohibir base->base, porque el solver podría
    # intercalar un cargador y dejar la ruta sin visitas.)
    if cfg.solver.require_all_vehicles_used and fleet_size > 1:
        def visit_count_cb(from_index: int) -> int:
            node = manager.IndexToNode(from_index)
            return 1 if node in visit_idxs else 0

        visit_cb_idx = routing.RegisterUnaryTransitCallback(visit_count_cb)
        routing.AddDimension(
            visit_cb_idx,
            0,                 # sin slack
            len(visits),       # cota superior (todas las visitas)
            True,              # cumul inicial = 0
            "visit_count",
        )
        visit_dim = routing.GetDimensionOrDie("visit_count")
        for v in range(fleet_size):
            end_index = routing.End(v)
            visit_dim.CumulVar(end_index).SetMin(1)

    # --- Dimensión de tiempo de conducción + makespan -----------------------
    # El tiempo de un arco = distancia / velocidad media. Se modela una
    # dimensión "time" (en segundos enteros). El makespan (tiempo del vehículo
    # que más tarda) se penaliza con SetGlobalSpanCostCoefficient sobre esa
    # dimensión: eso empuja a equilibrar la carga entre vehículos.
    #
    # El coeficiente convierte "segundos de span" a las unidades del objetivo
    # (energía escalada). A makespan_weight = 1, un segundo de span pesa como la
    # energía de conducir un segundo a la velocidad media; así el peso es una
    # perilla relativa adimensional e independiente del factor de escala.
    avg_speed_kmh = max(1e-6, float(cfg.solver.avg_speed_kmh))
    makespan_weight = float(cfg.solver.makespan_weight)

    def time_transit_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        km = float(sub_dist_km[i][j])
        seconds = km / avg_speed_kmh * 3600.0
        return int(round(seconds))

    time_cb_idx = routing.RegisterTransitCallback(time_transit_cb)
    # Horizonte amplio: suma de todos los arcos de la red como cota superior.
    max_seconds = int(round(float(sub_dist_km.sum()) / avg_speed_kmh * 3600.0)) + 1
    routing.AddDimension(
        time_cb_idx,
        0,            # sin espera/slack en los nodos
        max_seconds,  # cota superior del tiempo acumulado por vehículo
        True,         # tiempo inicial = 0 en la base
        "time",
    )
    time_dim = routing.GetDimensionOrDie("time")

    if makespan_weight > 0 and fleet_size > 1:
        # Energía (escalada) por segundo de conducción a la velocidad media:
        #   kWh/km * (km/s) = kWh/s ; * scale = micro-kWh/s.
        kwh_per_km = max(1e-9, model.kwh_per_km)
        km_per_s = avg_speed_kmh / 3600.0
        energy_per_second_scaled = kwh_per_km * km_per_s * scale
        coef = int(round(makespan_weight * energy_per_second_scaled))
        coef = max(1, coef)
        time_dim.SetGlobalSpanCostCoefficient(coef)

    # --- Búsqueda -----------------------------------------------------------
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = _enum_first_solution(
        cfg.solver.first_solution_strategy
    )
    search.local_search_metaheuristic = _enum_metaheuristic(
        cfg.solver.local_search_metaheuristic
    )
    search.time_limit.FromSeconds(cfg.solver.time_limit_s)
    search.log_search = False

    solution = routing.SolveWithParameters(search)

    if solution is None:
        return FleetResult(status="NO_SOLUTION", feasible=False)

    # --- Extraer una ruta por vehículo --------------------------------------
    routes: list[VehicleRoute] = []
    total_energy_scaled = 0
    makespan_seconds = 0
    for v in range(fleet_size):
        route_nodes: list[int] = []
        soc_values: list[float] = []
        index = routing.Start(v)
        route_energy_scaled = 0
        while True:
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            soc_values.append(solution.Value(soc_dim.CumulVar(index)) / scale)
            if routing.IsEnd(index):
                break
            next_index = solution.Value(routing.NextVar(index))
            arc = cost_cb(index, next_index)
            route_energy_scaled += arc
            index = next_index

        node_ids = [model_nodes[k].node_id for k in route_nodes]
        chargers_used = [
            model_nodes[k].node_id for k in route_nodes if k in charger_idxs
        ]
        n_visits = sum(1 for k in route_nodes if k in visit_idxs)
        total_energy_scaled += route_energy_scaled

        # Tiempo de conducción de la ruta = cumul final de la dimensión "time".
        route_seconds = solution.Value(time_dim.CumulVar(routing.End(v)))
        makespan_seconds = max(makespan_seconds, route_seconds)

        routes.append(
            VehicleRoute(
                vehicle_id=v,
                node_ids=node_ids,
                node_indices=[global_idx[k] for k in route_nodes],
                drive_energy_kwh=route_energy_scaled / scale,
                chargers_used=chargers_used,
                soc_at_nodes_kwh=soc_values,
                n_visits=n_visits,
                drive_time_hours=route_seconds / 3600.0,
            )
        )

    # --- Factibilidad: todas las visitas cubiertas exactamente una vez -------
    covered: list[str] = []
    for r in routes:
        covered.extend([nid for nid in r.node_ids if nid in {v.node_id for v in visits}])
    all_visits = {v.node_id for v in visits}
    covered_set = set(covered)
    all_covered = covered_set == all_visits and len(covered) == len(all_visits)

    status = "OK"
    if not all_covered:
        status = "MISSING_VISITS"

    return FleetResult(
        routes=routes,
        total_drive_energy_kwh=total_energy_scaled / scale,
        feasible=all_covered,
        status=status,
        makespan_drive_hours=makespan_seconds / 3600.0,
    )
