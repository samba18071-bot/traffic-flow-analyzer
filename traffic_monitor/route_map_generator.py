"""
Interactive Route Map and Traffic Dashboard Generator.
Generates an interactive, standalone HTML/SVG visualization of the road network,
live congestion statuses, camera sensor feeds, and recommended navigation routes.
"""

from typing import Dict, Any, List
import json


class RouteMapGenerator:
    """Generates an interactive web-based visual dashboard for route guidance."""

    @staticmethod
    def generate_html(
        engine: Any,
        comparison_results: Dict[str, Any],
        output_filepath: str = "route_guidance_dashboard.html"
    ) -> str:
        """
        Builds a self-contained HTML visualization with SVG road network topology,
        interactive route toggles, telemetry cards, and turn-by-turn navigation.
        """
        nodes_data = []
        for nid, data in engine.node_metadata.items():
            nodes_data.append({
                "id": nid,
                "name": data["name"],
                "x": data["x"] * 8.5 + 60,
                "y": 550 - (data["y"] * 5.5 + 40)
            })

        edges_data = []
        for u, v in engine.graph.edges():
            edge = engine.graph[u][v]
            u_node = next(n for n in nodes_data if n["id"] == u)
            v_node = next(n for n in nodes_data if n["id"] == v)
            los = edge.get("current_los", "A")

            # Color mapping per Level of Service
            los_colors = {
                "A": "#00E676",  # Green
                "B": "#69F0AE",  # Light Green
                "C": "#FFD600",  # Yellow
                "D": "#FF9100",  # Orange
                "E": "#FF6D00",  # Deep Orange
                "F": "#FF1744"   # Red
            }

            edges_data.append({
                "u": u,
                "v": v,
                "name": edge["name"],
                "x1": u_node["x"],
                "y1": u_node["y"],
                "x2": v_node["x"],
                "y2": v_node["y"],
                "length_km": edge["length_km"],
                "speed_kmh": edge.get("current_speed_kmh", edge["speed_limit_kmh"]),
                "los": los,
                "color": los_colors.get(los, "#00E676"),
                "lanes": edge["lanes"],
                "road_width_m": edge["road_width_m"],
                "camera_id": edge.get("camera_id"),
                "has_hazard": edge.get("has_hazard", False)
            })

        routes_json = json.dumps(comparison_results, indent=2)
        nodes_json = json.dumps(nodes_data, indent=2)
        edges_json = json.dumps(edges_data, indent=2)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🚦 Intelligent Traffic-Aware Dynamic Route Guidance System</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --primary: #38bdf8;
      --accent: #f59e0b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      padding: 14px 24px;
      background: #090d16;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    header h1 {{ font-size: 1.15rem; font-weight: 700; color: var(--primary); display: flex; align-items: center; gap: 8px; }}
    .badge {{
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}
    .layout {{ display: flex; flex: 1; overflow: hidden; }}
    .canvas-panel {{
      flex: 1;
      position: relative;
      background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    svg {{ width: 100%; height: 100%; max-width: 950px; max-height: 650px; }}
    .sidebar {{
      width: 440px;
      background: var(--card-bg);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }}
    .sidebar-section {{ padding: 20px; border-bottom: 1px solid var(--border); }}
    .route-selector {{ display: flex; gap: 8px; margin-top: 12px; }}
    .tab-btn {{
      flex: 1;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 600;
      font-size: 0.8rem;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: var(--primary);
      color: #0f172a;
      border-color: var(--primary);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-top: 14px;
    }}
    .metric-card {{
      background: #0f172a;
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 8px;
    }}
    .metric-title {{ font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-val {{ font-size: 1.25rem; font-weight: 700; color: #fff; margin-top: 4px; }}
    .steps-list {{ margin-top: 14px; display: flex; flex-direction: column; gap: 10px; }}
    .step-item {{
      background: #0f172a;
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 8px;
      font-size: 0.85rem;
    }}
    .step-header {{ display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 4px; }}
    .step-meta {{ color: var(--text-muted); font-size: 0.78rem; display: flex; gap: 12px; }}
    .legend {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--border);
      padding: 10px 14px;
      border-radius: 8px;
      backdrop-filter: blur(8px);
      display: flex;
      gap: 14px;
      font-size: 0.75rem;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  </style>
</head>
<body>
  <header>
    <h1>🚦 Traffic-Aware Dynamic Route Guidance System</h1>
    <div style="display:flex; gap:10px; align-items:center;">
      <span class="badge">Bureau of Public Roads (BPR) Engine</span>
      <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border-color: rgba(16, 185, 129, 0.3);">Real-Time CV Telemetry Active</span>
    </div>
  </header>

  <div class="layout">
    <div class="canvas-panel">
      <svg id="roadSvg" viewBox="0 0 950 600">
        <defs>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        <!-- Dynamic SVG content generated by JS -->
      </svg>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#00E676;"></div> LOS A/B (Free Flow)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FFD600;"></div> LOS C (Stable)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FF9100;"></div> LOS D (Capacity)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FF1744;"></div> LOS E/F (Bottleneck / Jam)</div>
      </div>
    </div>

    <div class="sidebar">
      <div class="sidebar-section">
        <div style="font-size:0.8rem; color:var(--text-muted);">RECOMMENDED DESTINATION GUIDANCE</div>
        <div style="font-size:1.1rem; font-weight:700; margin-top:4px;" id="routeTitle">Loading route...</div>
        
        <div class="route-selector">
          <button class="tab-btn active" onclick="switchRoute('fastest')">⚡ Fastest</button>
          <button class="tab-btn" onclick="switchRoute('eco')">🍃 Eco-Green</button>
          <button class="tab-btn" onclick="switchRoute('smooth')">🧘 Smoothest</button>
        </div>

        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-title">ESTIMATED TIME</div>
            <div class="metric-val" id="valEta">-- min</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">TOTAL DISTANCE</div>
            <div class="metric-val" id="valDist">-- km</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">AVERAGE SPEED</div>
            <div class="metric-val" id="valSpeed">-- km/h</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">FUEL CONSUMPTION</div>
            <div class="metric-val" id="valFuel">-- L</div>
          </div>
        </div>
      </div>

      <div class="sidebar-section" style="flex:1;">
        <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted);">
          Turn-by-Turn Navigation & Road Conditions
        </div>
        <div class="steps-list" id="stepsContainer"></div>
      </div>
    </div>
  </div>

  <script>
    const dataNodes = {nodes_json};
    const dataEdges = {edges_json};
    const routeComparison = {routes_json};
    let currentStrategy = 'fastest';

    function renderNetwork() {{
      const svg = document.getElementById('roadSvg');
      svg.innerHTML = '';

      const activeRoute = routeComparison.routes[currentStrategy];
      const activePath = activeRoute ? activeRoute.path_nodes : [];

      // Draw base road segments
      dataEdges.forEach(e => {{
        const isPath = isActiveEdge(e.u, e.v, activePath);
        const strokeW = isPath ? 7 : (e.lanes * 1.5 + 2);
        const strokeColor = isPath ? '#38bdf8' : e.color;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', e.x1);
        line.setAttribute('y1', e.y1);
        line.setAttribute('x2', e.x2);
        line.setAttribute('y2', e.y2);
        line.setAttribute('stroke', strokeColor);
        line.setAttribute('stroke-width', strokeW);
        line.setAttribute('stroke-linecap', 'round');
        if (isPath) line.setAttribute('filter', 'url(#glow)');
        svg.appendChild(line);

        // Road Name & Width Tag
        const mx = (e.x1 + e.x2) / 2;
        const my = (e.y1 + e.y2) / 2;
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', mx);
        text.setAttribute('y', my - 8);
        text.setAttribute('fill', '#94a3b8');
        text.setAttribute('font-size', '10');
        text.setAttribute('text-anchor', 'middle');
        text.textContent = `${{e.name}} (${{e.road_width_m}}m | ${{e.lanes}}L)`;
        svg.appendChild(text);

        // Camera Sensor Marker
        if (e.camera_id) {{
          const cam = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          cam.setAttribute('cx', mx);
          cam.setAttribute('cy', my);
          cam.setAttribute('r', '6');
          cam.setAttribute('fill', '#f59e0b');
          cam.setAttribute('stroke', '#fff');
          cam.setAttribute('stroke-width', '1.5');
          svg.appendChild(cam);
        }}
      }});

      // Draw intersection nodes
      dataNodes.forEach(n => {{
        const isOrigin = n.id === routeComparison.origin;
        const isDest = n.id === routeComparison.destination;
        const inPath = activePath.includes(n.id);

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', n.x);
        circle.setAttribute('cy', n.y);
        circle.setAttribute('r', isOrigin || isDest ? '12' : '8');
        circle.setAttribute('fill', isOrigin ? '#10b981' : (isDest ? '#ef4444' : (inPath ? '#38bdf8' : '#1e293b')));
        circle.setAttribute('stroke', '#fff');
        circle.setAttribute('stroke-width', '2');
        svg.appendChild(circle);

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', n.x);
        label.setAttribute('y', n.y + 24);
        label.setAttribute('fill', '#f8fafc');
        label.setAttribute('font-weight', '700');
        label.setAttribute('font-size', '11');
        label.setAttribute('text-anchor', 'middle');
        label.textContent = n.name;
        svg.appendChild(label);
      }});
    }}

    function isActiveEdge(u, v, path) {{
      for (let i = 0; i < path.length - 1; i++) {{
        if (path[i] === u && path[i + 1] === v) return true;
      }}
      return false;
    }}

    function updateSidebar() {{
      const r = routeComparison.routes[currentStrategy];
      if (!r) return;

      document.getElementById('routeTitle').textContent = `${{routeComparison.origin_name}} → ${{routeComparison.destination_name}}`;
      document.getElementById('valEta').textContent = `${{r.total_time_min}} min`;
      document.getElementById('valDist').textContent = `${{r.total_distance_km}} km`;
      document.getElementById('valSpeed').textContent = `${{r.avg_speed_kmh}} km/h`;
      document.getElementById('valFuel').textContent = `${{r.total_fuel_liters}} L`;

      const stepsContainer = document.getElementById('stepsContainer');
      stepsContainer.innerHTML = '';

      r.turn_by_turn.forEach((step, idx) => {{
        const div = document.createElement('div');
        div.className = 'step-item';
        div.innerHTML = `
          <div class="step-header">
            <span>${{idx + 1}}. ${{step.road_name}}</span>
            <span style="color:#38bdf8;">${{step.eta_min}}m</span>
          </div>
          <div style="font-size:0.8rem; color:#cbd5e1; margin-bottom:4px;">
            ${{step.from_name}} ➔ ${{step.to_name}}
          </div>
          <div class="step-meta">
            <span>📏 ${{step.distance_km}} km</span>
            <span>🛣️ ${{step.road_width_m}}m (${{step.lanes}} Lanes)</span>
            <span>🚗 ${{step.speed_kmh}} km/h</span>
            <span style="font-weight:700; color:${{step.los === 'A' || step.los === 'B' ? '#10b981' : (step.los === 'C' ? '#f59e0b' : '#ef4444')}}">
              LOS ${{step.los}}
            </span>
          </div>
        `;
        stepsContainer.appendChild(div);
      }});
    }}

    function switchRoute(strat) {{
      currentStrategy = strat;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.toLowerCase().includes(strat));
      if (activeBtn) activeBtn.classList.add('active');
      renderNetwork();
      updateSidebar();
    }}

    // Initial render
    renderNetwork();
    updateSidebar();
  </script>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_filepath

