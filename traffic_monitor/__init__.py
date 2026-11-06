"""
Traffic Flow Analyzer
A real-time intelligent vehicle detection, tracking, counting, velocity estimation,
road width measurement, and traffic safety analytics system.
"""

from traffic_monitor.road_width_estimator import RoadWidthEstimator
from traffic_monitor.traffic_analytics import TrafficAnalytics
from traffic_monitor.routing_engine import DynamicRoutingEngine
from traffic_monitor.route_map_generator import RouteMapGenerator

__all__ = [
    "RoadWidthEstimator",
    "TrafficAnalytics",
    "DynamicRoutingEngine",
    "RouteMapGenerator"
]
__version__ = "2.0.0"
__author__ = "Samba"


