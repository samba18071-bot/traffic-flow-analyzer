"""
Unit tests for DynamicRoutingEngine and BPR Link Cost functions.
"""

import os
from traffic_monitor.routing_engine import DynamicRoutingEngine
from traffic_monitor.route_map_generator import RouteMapGenerator


def test_default_network():
    engine = DynamicRoutingEngine()
    assert len(engine.graph.nodes) >= 6
    assert len(engine.graph.edges) >= 10
    assert "A" in engine.graph
    assert "F" in engine.graph


def test_free_flow_fastest_route():
    engine = DynamicRoutingEngine()
    route = engine.find_route(origin="A", destination="F", strategy="fastest")
    assert route is not None
    assert route["path_nodes"][0] == "A"
    assert route["path_nodes"][-1] == "F"
    assert route["total_time_min"] > 0
    assert route["total_distance_km"] > 0
    # In normal conditions with highway speeds, A -> B -> D -> F is the primary arterial
    assert "B" in route["path_nodes"]


def test_dynamic_rerouting_on_congestion():
    engine = DynamicRoutingEngine()

    # Before congestion: route goes through A -> B
    route_initial = engine.find_route("A", "F", strategy="fastest")
    assert route_initial["path_nodes"][1] == "B"

    # Inject heavy congestion + stalled truck incident on A -> B
    engine.update_traffic_telemetry(
        u="A",
        v="B",
        vehicle_volume=180,
        avg_speed_kmh=8.0,
        los_grade="F",
        has_hazard=True
    )

    # After congestion: system should automatically reroute via Coastal Corridor (A -> C)
    route_rerouted = engine.find_route("A", "F", strategy="fastest")
    assert route_rerouted is not None
    assert route_rerouted["path_nodes"][1] == "C", f"Expected detour via C, got {route_rerouted['path_nodes']}"


def test_bpr_cost_scaling():
    engine = DynamicRoutingEngine()
    cost_empty = engine.calculate_edge_cost("A", "B", strategy="fastest")

    # High volume / near capacity
    engine.update_traffic_telemetry("A", "B", vehicle_volume=120, avg_speed_kmh=25.0)
    cost_busy = engine.calculate_edge_cost("A", "B", strategy="fastest")

    assert cost_busy > cost_empty


def test_compare_routes_and_html_generation(tmp_path):
    engine = DynamicRoutingEngine()
    comparison = engine.compare_routes("A", "F")
    assert "fastest" in comparison["routes"]
    assert "eco" in comparison["routes"]
    assert "smooth" in comparison["routes"]

    html_path = str(tmp_path / "test_route_dashboard.html")
    out = RouteMapGenerator.generate_html(engine, comparison, output_filepath=html_path)
    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Traffic-Aware Dynamic Route Guidance" in content
    assert "svg" in content

