"""
Dynamic Traffic Routing Engine.
Calculates optimal routes across a road network using real-time traffic telemetry,
road width geometry, Bureau of Public Roads (BPR) congestion functions,
and multi-objective Pareto optimization (Fastest, Eco-Friendly, Smoothest).
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import networkx as nx


class DynamicRoutingEngine:
    """
    Intelligent Route Guidance Engine combining live computer vision traffic telemetry
    with graph network optimization.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self._init_default_network()

    def add_intersection(self, node_id: str, name: str, x: float, y: float):
        """Adds a road junction/intersection to the network graph."""
        self.graph.add_node(node_id, name=name, x=x, y=y)
        self.node_metadata[node_id] = {"name": name, "x": x, "y": y}

    def add_road_segment(
        self,
        u: str,
        v: str,
        name: str,
        length_km: float,
        speed_limit_kmh: float = 60.0,
        lanes: int = 2,
        road_width_m: float = 7.0,
        capacity_per_lane_vph: float = 1800.0,
        bidirectional: bool = False
    ):
        """
        Adds a directed road segment with physical carriageway parameters.
        """
        edge_data = {
            "name": name,
            "length_km": max(0.05, float(length_km)),
            "speed_limit_kmh": float(speed_limit_kmh),
            "lanes": max(1, int(lanes)),
            "road_width_m": max(3.0, float(road_width_m)),
            "capacity_vph": float(lanes * capacity_per_lane_vph),
            # Dynamic traffic state
            "current_volume": 0,
            "current_speed_kmh": float(speed_limit_kmh),
            "current_los": "A",
            "has_hazard": False,
            "camera_id": None
        }

        self.graph.add_edge(u, v, **edge_data)
        if bidirectional:
            self.graph.add_edge(v, u, **dict(edge_data))

    def bind_camera_to_segment(self, u: str, v: str, camera_id: str):
        """Associates a computer vision camera feed with a specific road segment."""
        if self.graph.has_edge(u, v):
            self.graph[u][v]["camera_id"] = camera_id

    def update_traffic_telemetry(
        self,
        u: str,
        v: str,
        vehicle_volume: int,
        avg_speed_kmh: float,
        road_width_m: Optional[float] = None,
        lanes: Optional[int] = None,
        los_grade: Optional[str] = None,
        has_hazard: bool = False
    ):
        """
        Injects real-time computer vision metrics directly into the road graph edge.
        """
        if not self.graph.has_edge(u, v):
            return

        edge = self.graph[u][v]
        edge["current_volume"] = int(vehicle_volume)
        edge["current_speed_kmh"] = max(1.0, float(avg_speed_kmh))
        edge["has_hazard"] = bool(has_hazard)

        if road_width_m is not None and road_width_m > 0:
            edge["road_width_m"] = float(road_width_m)
        if lanes is not None and lanes > 0:
            edge["lanes"] = int(lanes)
            edge["capacity_vph"] = edge["lanes"] * 1800.0
        if los_grade:
            edge["current_los"] = los_grade

    def calculate_edge_cost(self, u: str, v: str, strategy: str = "fastest") -> float:
        """
        Computes dynamic edge cost using the Bureau of Public Roads (BPR) function
        and specific routing objective weights.
        """
        edge = self.graph[u][v]
        length_km = edge["length_km"]
        speed_limit = edge["speed_limit_kmh"]
        measured_speed = edge.get("current_speed_kmh", speed_limit)
        volume = edge.get("current_volume", 0)
        capacity = edge.get("capacity_vph", 3600.0)
        has_hazard = edge.get("has_hazard", False)
        los = edge.get("current_los", "A")

        # 1. Free-flow travel time (minutes)
        t0_min = (length_km / speed_limit) * 60.0

        # 2. BPR Link Performance Function: T = T0 * (1 + alpha * (V / C)^beta)
        # Scaled volume-to-capacity ratio for corridor simulation
        voc_ratio = (volume * 60.0) / max(capacity, 1.0)
        alpha, beta = 0.15, 4.0
        bpr_time_min = t0_min * (1.0 + alpha * math.pow(min(voc_ratio, 3.0), beta))

        # Blend with empirical measured speed from computer vision
        empirical_time_min = (length_km / max(measured_speed, 1.0)) * 60.0
        travel_time_min = 0.5 * bpr_time_min + 0.5 * empirical_time_min

        # 3. Hazard and Bottleneck Penalties
        hazard_penalty = 15.0 if has_hazard else 0.0

        # Cost strategy weighting
        if strategy == "fastest":
            # Minimize absolute travel time (minutes)
            cost = travel_time_min + hazard_penalty

        elif strategy == "eco":
            # Fuel consumption model (Liters):
            # Cruising at 50-70 km/h: ~0.06 L/km. Severe stop-and-go (<20 km/h or LOS E/F): ~0.16 L/km
            if measured_speed >= 50.0:
                fuel_rate_l_km = 0.065
            elif measured_speed >= 30.0:
                fuel_rate_l_km = 0.095
            else:
                fuel_rate_l_km = 0.160
            fuel_liters = length_km * fuel_rate_l_km
            # Weighted combination of fuel and travel time
            cost = fuel_liters * 10.0 + travel_time_min * 0.5 + hazard_penalty

        elif strategy == "smooth":
            # Penalize congestion grades heavily to ensure comfortable, low-stress driving
            los_penalties = {"A": 0.0, "B": 1.0, "C": 3.0, "D": 7.0, "E": 14.0, "F": 25.0}
            los_weight = los_penalties.get(los, 0.0)
            cost = travel_time_min + los_weight + (25.0 if has_hazard else 0.0)

        else:
            # Balanced strategy
            cost = travel_time_min + (0.5 * hazard_penalty)

        return max(0.01, float(cost))

    def find_route(
        self,
        origin: str,
        destination: str,
        strategy: str = "fastest"
    ) -> Optional[Dict[str, Any]]:
        """
        Computes the optimal path from origin to destination under the given strategy.
        """
        if origin not in self.graph or destination not in self.graph:
            return None

        # Build weighted weight attribute on graph edges
        for u, v in self.graph.edges():
            self.graph[u][v]["weight"] = self.calculate_edge_cost(u, v, strategy)

        try:
            path = nx.shortest_path(self.graph, source=origin, target=destination, weight="weight")
            return self._compile_route_details(path, strategy)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def compare_routes(
        self,
        origin: str,
        destination: str
    ) -> Dict[str, Any]:
        """
        Calculates and compares candidate routes across all key objectives:
        Fastest (Min Time), Eco-Friendly (Min Emissions/Fuel), and Smoothest (Min Congestion).
        """
        results = {}
        for strategy in ["fastest", "eco", "smooth"]:
            route = self.find_route(origin, destination, strategy)
            if route:
                results[strategy] = route

        # Identify candidate alternatives
        unique_paths = []
        filtered_results = {}
        for strat, r in results.items():
            path_tuple = tuple(r["path_nodes"])
            if path_tuple not in unique_paths:
                unique_paths.append(path_tuple)
                filtered_results[strat] = r
            else:
                filtered_results[strat] = r

        return {
            "origin": origin,
            "origin_name": self.node_metadata.get(origin, {}).get("name", origin),
            "destination": destination,
            "destination_name": self.node_metadata.get(destination, {}).get("name", destination),
            "routes": filtered_results
        }

    def _compile_route_details(self, path: List[str], strategy: str) -> Dict[str, Any]:
        """Aggregates comprehensive metrics and turn-by-turn steps along a path."""
        total_distance_km = 0.0
        total_time_min = 0.0
        total_fuel_l = 0.0
        bottlenecks = []
        turn_by_turn = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = self.graph[u][v]
            length = edge["length_km"]
            speed = edge.get("current_speed_kmh", edge["speed_limit_kmh"])
            los = edge.get("current_los", "A")
            has_hazard = edge.get("has_hazard", False)

            time_min = (length / max(speed, 1.0)) * 60.0
            fuel_rate = 0.065 if speed >= 50 else (0.095 if speed >= 30 else 0.16)
            fuel_l = length * fuel_rate

            total_distance_km += length
            total_time_min += time_min
            total_fuel_l += fuel_l

            u_name = self.node_metadata.get(u, {}).get("name", u)
            v_name = self.node_metadata.get(v, {}).get("name", v)

            if los in ["D", "E", "F"] or has_hazard:
                bottlenecks.append({
                    "segment": f"{u_name} → {v_name}",
                    "los": los,
                    "speed_kmh": round(speed, 1),
                    "hazard": has_hazard
                })

            turn_by_turn.append({
                "from_node": u,
                "from_name": u_name,
                "to_node": v,
                "to_name": v_name,
                "road_name": edge["name"],
                "distance_km": round(length, 2),
                "eta_min": round(time_min, 1),
                "speed_kmh": round(speed, 1),
                "los": los,
                "lanes": edge["lanes"],
                "road_width_m": edge["road_width_m"]
            })

        # Calculate estimated CO2 footprint (2.31 kg CO2 per liter of gasoline)
        co2_kg = total_fuel_l * 2.31
        avg_speed_kmh = (total_distance_km / (total_time_min / 60.0)) if total_time_min > 0 else 0.0

        return {
            "strategy": strategy,
            "path_nodes": path,
            "path_names": [self.node_metadata.get(n, {}).get("name", n) for n in path],
            "total_distance_km": round(total_distance_km, 2),
            "total_time_min": round(total_time_min, 1),
            "avg_speed_kmh": round(avg_speed_kmh, 1),
            "total_fuel_liters": round(total_fuel_l, 2),
            "co2_emissions_kg": round(co2_kg, 2),
            "bottlenecks": bottlenecks,
            "turn_by_turn": turn_by_turn
        }

    def _init_default_network(self):
        """Initializes a realistic benchmark urban corridor network for demonstration and testing."""
        # Nodes (Intersections & Landmarks)
        nodes = [
            ("A", "Downtown Central", 10.0, 50.0),
            ("B", "Tech Park Junction", 30.0, 75.0),
            ("C", "Harbor Expressway", 35.0, 25.0),
            ("D", "Metro Ring Interchange", 60.0, 60.0),
            ("E", "Old Town Boulevard", 65.0, 20.0),
            ("F", "International Airport", 90.0, 45.0)
        ]
        for nid, name, x, y in nodes:
            self.add_intersection(nid, name, x, y)

        # Edges (Road Segments)
        # Route 1: Direct Highway Arterial (Fast when clear, but prone to bottlenecks)
        self.add_road_segment("A", "B", "Main Highway North", length_km=4.2, speed_limit_kmh=70.0, lanes=3, road_width_m=10.5, bidirectional=True)
        self.add_road_segment("B", "D", "Central Expressway", length_km=5.0, speed_limit_kmh=80.0, lanes=4, road_width_m=14.0, bidirectional=True)
        self.add_road_segment("D", "F", "Airport Bypass Flyover", length_km=4.8, speed_limit_kmh=80.0, lanes=4, road_width_m=14.0, bidirectional=True)

        # Route 2: Scenic Waterfront Corridor (Longer, steady flow, highly eco-friendly)
        self.add_road_segment("A", "C", "Harbor Shore Parkway", length_km=3.8, speed_limit_kmh=60.0, lanes=2, road_width_m=7.0, bidirectional=True)
        self.add_road_segment("C", "E", "Coastal Highway Link", length_km=4.5, speed_limit_kmh=65.0, lanes=2, road_width_m=7.5, bidirectional=True)
        self.add_road_segment("E", "F", "Airport South Access Rd", length_km=5.2, speed_limit_kmh=60.0, lanes=2, road_width_m=7.0, bidirectional=True)

        # Connectors / Cross-Links (Enabling smart dynamic rerouting)
        self.add_road_segment("B", "C", "Crosstown Avenue", length_km=3.2, speed_limit_kmh=50.0, lanes=2, road_width_m=7.0, bidirectional=True)
        self.add_road_segment("C", "D", "Metro Link Diagonal", length_km=4.1, speed_limit_kmh=55.0, lanes=2, road_width_m=7.0, bidirectional=True)
        self.add_road_segment("D", "E", "River Crossing Causeway", length_km=3.6, speed_limit_kmh=60.0, lanes=2, road_width_m=7.5, bidirectional=True)

        # Associate vision camera sensors with main corridor links
        self.bind_camera_to_segment("A", "B", camera_id="CAM_HWY_01")
        self.bind_camera_to_segment("B", "D", camera_id="CAM_HWY_02")
        self.bind_camera_to_segment("C", "E", camera_id="CAM_CST_01")
