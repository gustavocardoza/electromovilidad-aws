"""EVRP solver package.

Resolución de un Electric Vehicle Routing Problem (EVRP) con minimización de
energía y restricción dura de estado de carga (SOC) sobre instancias de nodos
de ruteo. Enfoque híbrido: OR-Tools decide la secuencia de nodos y los
cargadores a usar (SOC como restricción dura), y un post-proceso determinista
calcula la recarga parcial realista por potencia de estación.
"""
