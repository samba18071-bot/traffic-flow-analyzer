"""
Route Navigator CLI & Demonstration runner.
Demonstrates dynamic route calculation, traffic telemetry injection,
corridor bottleneck rerouting, and interactive HTML dashboard generation.
"""

import argparse
import sys
from traffic_monitor.routing_engine import DynamicRoutingEngine
from traffic_monitor.route_map_generator import RouteMapGenerator


def run_navigation(args):
    engine = DynamicRoutingEngine()

    print("\n========================================================")
    print("🚦 INTELLIGENT TRAFFIC-AWARE DYNAMIC ROUTE GUIDANCE")
    print("========================================================")
    print(f"Network: {len(engine.graph.nodes)} Junctions | {len(engine.graph.edges)} Directed Road Links")

    # If simulation flag enabled, inject synthetic congestion scenario
    if args.simulate_congestion:
        print("\n[SCENARIO] Simulating heavy congestion bottleneck on Highway Link (A → B):")
        print("  - Detected Vehicle Count: 145 vehicles")
        print("  - Observed Speed: 14.5 km/h (Gridlock)")
        print("  - Level of Service: LOS F")
        engine.update_traffic_telemetry(
            u="A",
            v="B",
            vehicle_volume=145,
            avg_speed_kmh=14.5,
            road_width_m=10.5,
            lanes=3,
            los_grade="F",
            has_hazard=True
        )
        print("  - Injected Incident: STALLED TRUCK / HAZARD on Link A → B")

    print(f"\n[QUERY] Searching optimal route from [{args.origin}] to [{args.destination}]...")
    comparison = engine.compare_routes(args.origin, args.destination)

    if not comparison["routes"]:
        print(f"[ERROR] No valid path found between {args.origin} and {args.destination}.")
        sys.exit(1)

    print(f"\n📍 ORIGIN:      {comparison['origin_name']} ({args.origin})")
    print(f"🏁 DESTINATION: {comparison['destination_name']} ({args.destination})")
    print("-" * 65)

    for strat, r in comparison["routes"].items():
        title = strat.upper()
        print(f"\n▶ STRATEGY: {title}")
        print(f"  • Path:         {' ➔ '.join(r['path_names'])}")
        print(f"  • Total Time:   {r['total_time_min']} mins")
        print(f"  • Distance:     {r['total_distance_km']} km")
        print(f"  • Avg Speed:    {r['avg_speed_kmh']} km/h")
        print(f"  • Fuel Usage:   {r['total_fuel_liters']} Liters (~{r['co2_emissions_kg']} kg CO2)")
        if r["bottlenecks"]:
            print(f"  ⚠️  Bottlenecks ({len(r['bottlenecks'])}):")
            for b in r["bottlenecks"]:
                print(f"     - {b['segment']} | LOS {b['los']} | {b['speed_kmh']} km/h")

    # Generate interactive HTML dashboard
    if args.html:
        out_file = RouteMapGenerator.generate_html(engine, comparison, output_filepath=args.html)
        print(f"\n[SUCCESS] Generated Interactive Visual Dashboard: {out_file}")
        print("  Open this file in your browser to inspect the live road network map and routes!")

    print("\n========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Traffic-Aware Dynamic Route Guidance System")
    parser.add_argument("--origin", type=str, default="A", help="Origin junction node ID (e.g. A, B, C, D, E, F)")
    parser.add_argument("--destination", type=str, default="F", help="Destination junction node ID")
    parser.add_argument("--simulate-congestion", action="store_true", help="Simulate real-time camera-detected bottleneck on primary corridor")
    parser.add_argument("--html", type=str, default="route_guidance_dashboard.html", help="Path to generate standalone interactive HTML map")

    args = parser.parse_args()
    run_navigation(args)


if __name__ == "__main__":
    main()
