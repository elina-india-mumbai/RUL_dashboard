"""
Water Pump Infrastructure Risk Analytics Dashboard
=====================================================
Streamlit app connecting to Neo4j Knowledge Graph.

Prerequisites:
    pip install streamlit neo4j pyvis networkx plotly pandas

Usage:
    streamlit run risk_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from neo4j import GraphDatabase

# ── NEO4J CONFIG ────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password_here"  # ← CHANGE THIS
# ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Water Pump Risk Analytics",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    .risk-card {
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border-left: 5px solid;
    }

    .risk-critical { background: #fff5f5; border-left-color: #e53e3e; }
    .risk-high { background: #fffaf0; border-left-color: #dd6b20; }
    .risk-medium { background: #fffff0; border-left-color: #d69e2e; }
    .risk-low { background: #f0fff4; border-left-color: #38a169; }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .fault-path {
        background: #f7fafc;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border: 1px solid #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .impact-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ── NEO4J CONNECTION ────────────────────────────────────────────

@st.cache_resource
def get_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        return None


def run_query(query, params=None):
    driver = get_driver()
    if driver is None:
        return []
    with driver.session() as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]


# ── RISK COMPUTATION ────────────────────────────────────────────

def compute_component_health(vce_on_current, vce_on_initial=1.3, vce_on_threshold=4.5):
    """Convert Vce_on value to 0-100 health index."""
    if vce_on_current <= vce_on_initial:
        return 100.0
    if vce_on_current >= vce_on_threshold:
        return 0.0
    return max(0, 100.0 * (1 - (vce_on_current - vce_on_initial) / (vce_on_threshold - vce_on_initial)))


def compute_esr_health(esr_current, esr_initial=15, esr_threshold=30):
    """Convert ESR to health index."""
    if esr_current <= esr_initial:
        return 100.0
    if esr_current >= esr_threshold:
        return 0.0
    return max(0, 100.0 * (1 - (esr_current - esr_initial) / (esr_threshold - esr_initial)))


def compute_system_risk(component_healths):
    """
    Compute system risk from component healths.
    Uses weakest-link + weighted combination.
    Risk = 100 - health (inverted scale: 0=safe, 100=critical)
    """
    weights = {
        "IGBT Module": 0.30,
        "DC-Link Capacitor": 0.25,
        "Pump Bearings": 0.20,
        "Mechanical Seal": 0.10,
        "Gate Driver": 0.05,
        "Stator Winding": 0.05,
        "Impeller": 0.03,
        "Diode Bridge Rectifier": 0.02,
    }
    # Weighted average risk
    weighted_risk = sum((100 - h) * weights.get(name, 0.05) for name, h in component_healths.items())
    # Weakest link penalty
    min_health = min(component_healths.values())
    weakest_link_risk = 100 - min_health
    # Combined: 60% weighted + 40% weakest link
    system_risk = 0.6 * weighted_risk + 0.4 * weakest_link_risk
    return min(100, max(0, system_risk))


def risk_to_color(risk):
    if risk >= 75:
        return "#e53e3e"
    elif risk >= 50:
        return "#dd6b20"
    elif risk >= 25:
        return "#d69e2e"
    else:
        return "#38a169"


def risk_to_label(risk):
    if risk >= 75:
        return "CRITICAL"
    elif risk >= 50:
        return "HIGH"
    elif risk >= 25:
        return "MODERATE"
    else:
        return "LOW"


def health_to_color(health):
    if health >= 70:
        return "#38a169"
    elif health >= 30:
        return "#d69e2e"
    else:
        return "#e53e3e"


# ── SIDEBAR ─────────────────────────────────────────────────────

st.sidebar.markdown("## 🔧 Component Health Inputs")
st.sidebar.markdown("Adjust sensor readings to simulate degradation scenarios.")
st.sidebar.markdown("---")

st.sidebar.markdown("### ⚡ Power Electronics")
vce_on = st.sidebar.slider("IGBT Vce_on (V)", 1.0, 5.0, 1.3, 0.1,
                            help="Collector-emitter saturation voltage. Healthy: 1.2-1.5V")
esr = st.sidebar.slider("Capacitor ESR (mΩ)", 10, 60, 15, 1,
                         help="Equivalent series resistance. Healthy: 10-20 mΩ")
gate_health = st.sidebar.slider("Gate Driver Health (%)", 0, 100, 95, 5)

st.sidebar.markdown("### 🔩 Mechanical")
bearing_vib = st.sidebar.slider("Bearing Vibration (mm/s)", 0.0, 15.0, 2.0, 0.5,
                                 help="ISO 10816: <4.5 = Good, 4.5-11.2 = Alert, >11.2 = Danger")
seal_health = st.sidebar.slider("Mechanical Seal Health (%)", 0, 100, 90, 5)

st.sidebar.markdown("### 🏭 Station Selection")
station = st.sidebar.selectbox("Pump Station",
                                ["Pump Station North", "Pump Station South", "Pump Station Central"])

# Compute healths
component_healths = {
    "IGBT Module": compute_component_health(vce_on),
    "DC-Link Capacitor": compute_esr_health(esr),
    "Gate Driver": float(gate_health),
    "Pump Bearings": max(0, 100 - (bearing_vib / 15.0) * 100),
    "Mechanical Seal": float(seal_health),
    "Stator Winding": 85.0,
    "Impeller": 90.0,
    "Diode Bridge Rectifier": 95.0,
}

system_risk = compute_system_risk(component_healths)

# ── MAIN DASHBOARD ──────────────────────────────────────────────

st.markdown('<p class="main-title">Water Pump Infrastructure Risk Analytics</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Knowledge Graph-Driven Prognostics | {station}</p>', unsafe_allow_html=True)

# ── ROW 1: Key Metrics ──
col1, col2, col3, col4 = st.columns(4)

with col1:
    risk_color = risk_to_color(system_risk)
    st.metric("System Risk Score", f"{system_risk:.0f}/100", risk_to_label(system_risk))

with col2:
    min_health_component = min(component_healths, key=component_healths.get)
    min_health = component_healths[min_health_component]
    st.metric("Weakest Component", min_health_component, f"{min_health:.0f}%")

with col3:
    # Estimated cost exposure
    if system_risk >= 75:
        cost_exposure = 350000
    elif system_risk >= 50:
        cost_exposure = 120000
    elif system_risk >= 25:
        cost_exposure = 30000
    else:
        cost_exposure = 5000
    st.metric("Cost Exposure", f"${cost_exposure:,.0f}", "If failure occurs")

with col4:
    # Population at risk
    station_pop = {"Pump Station North": 15000, "Pump Station South": 22000, "Pump Station Central": 35000}
    pop = station_pop.get(station, 15000)
    pop_at_risk = int(pop * system_risk / 100)
    st.metric("Population at Risk", f"{pop_at_risk:,}", f"of {pop:,} served")

st.markdown("---")

# ── ROW 2: Risk Gauge + Component Health ──
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📊 System Risk Gauge")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=system_risk,
        number={"suffix": "%", "font": {"size": 48, "family": "JetBrains Mono"}},
        title={"text": f"Overall Risk — {station}", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": risk_color, "thickness": 0.75},
            "bgcolor": "#f7fafc",
            "steps": [
                {"range": [0, 25], "color": "#f0fff4"},
                {"range": [25, 50], "color": "#fffff0"},
                {"range": [50, 75], "color": "#fffaf0"},
                {"range": [75, 100], "color": "#fff5f5"},
            ],
            "threshold": {
                "line": {"color": "#1a1a2e", "width": 3},
                "thickness": 0.8,
                "value": system_risk,
            },
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30),
                            paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_right:
    st.markdown("### 🔋 Component Health Index")

    # Horizontal bar chart
    health_df = pd.DataFrame([
        {"Component": name, "Health": health, "Color": health_to_color(health)}
        for name, health in sorted(component_healths.items(), key=lambda x: x[1])
    ])

    fig_health = go.Figure()
    for _, row in health_df.iterrows():
        fig_health.add_trace(go.Bar(
            y=[row["Component"]],
            x=[row["Health"]],
            orientation='h',
            marker_color=row["Color"],
            text=f'{row["Health"]:.0f}%',
            textposition='inside',
            textfont=dict(color='white', size=13, family='JetBrains Mono'),
            showlegend=False,
        ))
    fig_health.update_layout(
        xaxis=dict(range=[0, 105], title="Health Index (%)"),
        yaxis=dict(autorange="reversed"),
        height=300, margin=dict(t=10, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    st.plotly_chart(fig_health, use_container_width=True)

st.markdown("---")

# ── ROW 3: Fault Propagation Paths ──
col_fault, col_impact = st.columns([1, 1])

with col_fault:
    st.markdown("### 🔗 Active Fault Propagation Paths")
    st.markdown("*Causal chains from degraded components to service impact:*")

    # Build fault paths based on current health values
    active_paths = []

    if component_healths["IGBT Module"] < 50:
        active_paths.append({
            "path": "IGBT → Bond Wire Lift-off → Vce_on Rise → Conduction Loss ↑ → VFD Overtemp → Station Offline",
            "risk": 100 - component_healths["IGBT Module"],
            "component": "IGBT Module",
        })

    if component_healths["DC-Link Capacitor"] < 50:
        active_paths.append({
            "path": "Capacitor → ESR Drift → DC Bus Ripple → VFD DC Bus Fault → Station Offline",
            "risk": 100 - component_healths["DC-Link Capacitor"],
            "component": "DC-Link Capacitor",
        })

    if component_healths["Pump Bearings"] < 50:
        active_paths.append({
            "path": "Bearings → Wear → Vibration ↑ → Pump Seizure → Boil Water Advisory",
            "risk": 100 - component_healths["Pump Bearings"],
            "component": "Pump Bearings",
        })

    if component_healths["Mechanical Seal"] < 50:
        active_paths.append({
            "path": "Seal → Deterioration → Leakage → Water Ingress → Motor Failure → Emergency Distribution",
            "risk": 100 - component_healths["Mechanical Seal"],
            "component": "Mechanical Seal",
        })

    if component_healths["IGBT Module"] < 70 and component_healths["DC-Link Capacitor"] < 70:
        active_paths.append({
            "path": "IGBT + Capacitor → Combined VFD Degradation → Harmonic Distortion → Motor Insulation Failure → Boil Water Advisory",
            "risk": max(100 - component_healths["IGBT Module"], 100 - component_healths["DC-Link Capacitor"]) * 1.2,
            "component": "Multiple",
        })

    if not active_paths:
        st.success("✅ No active fault propagation paths. All components within healthy limits.")
    else:
        active_paths.sort(key=lambda x: x["risk"], reverse=True)
        for p in active_paths:
            risk_class = "risk-critical" if p["risk"] >= 75 else "risk-high" if p["risk"] >= 50 else "risk-medium"
            st.markdown(f'''
            <div class="risk-card {risk_class}">
                <div style="font-weight:600; margin-bottom:4px;">Risk: {p["risk"]:.0f}% — {p["component"]}</div>
                <div class="fault-path">{p["path"]}</div>
            </div>
            ''', unsafe_allow_html=True)

with col_impact:
    st.markdown("### 💰 Impact Estimation")

    if system_risk < 25:
        st.markdown("""
        <div style="background: #f0fff4; border-radius:12px; padding:20px; border:1px solid #c6f6d5;">
            <h4 style="color:#38a169; margin:0;">Low Risk — Normal Operations</h4>
            <p style="color:#666;">All components within acceptable limits. Continue routine monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Calculate impacts
        failure_prob = min(system_risk / 100, 0.95)
        repair_cost = sum(
            (100 - h) / 100 * {"IGBT Module": 2500, "DC-Link Capacitor": 800,
                                "Pump Bearings": 1200, "Mechanical Seal": 900,
                                "Gate Driver": 350, "Stator Winding": 5000,
                                "Impeller": 600, "Diode Bridge Rectifier": 200}.get(name, 500)
            for name, h in component_healths.items() if h < 70
        )
        downtime = max(4, int(system_risk / 100 * 72))
        reg_penalty = int(system_risk / 100 * 100000) if system_risk > 60 else 0
        total_cost = repair_cost + reg_penalty + (pop_at_risk * 5)  # $5/person emergency cost

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #2d3748 0%, #1a202c 100%); border-radius:12px; padding:20px; color:white;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Failure Probability</div>
                    <div style="font-size:2rem; font-weight:700; font-family:'JetBrains Mono'; color:{risk_color};">{failure_prob*100:.0f}%</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Expected Downtime</div>
                    <div style="font-size:2rem; font-weight:700; font-family:'JetBrains Mono';">{downtime}h</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Repair Cost</div>
                    <div style="font-size:2rem; font-weight:700; font-family:'JetBrains Mono';">${repair_cost:,.0f}</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Regulatory Penalty</div>
                    <div style="font-size:2rem; font-weight:700; font-family:'JetBrains Mono'; color:#fc8181;">${reg_penalty:,.0f}</div>
                </div>
            </div>
            <div style="margin-top:16px; border-top:1px solid #4a5568; padding-top:12px;">
                <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Total Risk-Weighted Cost Exposure</div>
                <div style="font-size:2.5rem; font-weight:700; font-family:'JetBrains Mono'; color:#fbd38d;">${total_cost:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── ROW 4: Maintenance Priority + RUL Timeline ──
col_maint, col_rul = st.columns([1, 1])

with col_maint:
    st.markdown("### 🛠️ Maintenance Priority Ranking")

    maint_data = []
    for name, health in component_healths.items():
        cost_map = {"IGBT Module": 2500, "DC-Link Capacitor": 800, "Pump Bearings": 1200,
                    "Mechanical Seal": 900, "Gate Driver": 350, "Stator Winding": 5000,
                    "Impeller": 600, "Diode Bridge Rectifier": 200}
        lead_map = {"IGBT Module": 14, "DC-Link Capacitor": 7, "Pump Bearings": 10,
                    "Mechanical Seal": 7, "Gate Driver": 5, "Stator Winding": 21,
                    "Impeller": 14, "Diode Bridge Rectifier": 3}
        urgency = (100 - health)
        roi = urgency / max(cost_map.get(name, 500), 1) * 1000
        maint_data.append({
            "Component": name,
            "Health": f"{health:.0f}%",
            "Urgency": f"{urgency:.0f}",
            "Cost": f"${cost_map.get(name, 500):,}",
            "Lead Time": f"{lead_map.get(name, 7)}d",
            "Action": "🔴 Replace Now" if health < 20 else "🟡 Schedule" if health < 50 else "🟢 Monitor",
        })

    maint_df = pd.DataFrame(maint_data)
    maint_df = maint_df.sort_values("Urgency", ascending=False, key=lambda x: x.astype(float))
    st.dataframe(maint_df, use_container_width=True, hide_index=True, height=320)

with col_rul:
    st.markdown("### 📈 RUL Degradation Projection")

    # Show sigmoid projection for IGBT
    L, k, t0, c = 4.456, 11.084, 0.0778, 0.0
    t = np.linspace(0, 0.8, 200)
    vce_trajectory = L / (1 + np.exp(-k * (t - t0))) + c
    threshold = c + 0.95 * L

    # Find current position
    current_pos = None
    for i, v in enumerate(vce_trajectory):
        if v >= vce_on:
            current_pos = t[i]
            break
    if current_pos is None:
        current_pos = t[-1]

    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(x=t, y=vce_trajectory, mode='lines',
                                  name='Predicted Degradation',
                                  line=dict(color='#3182ce', width=2.5)))
    fig_rul.add_hline(y=threshold, line_dash="dash", line_color="#e53e3e",
                       annotation_text=f"Failure Threshold ({threshold:.2f}V)")
    fig_rul.add_trace(go.Scatter(x=[current_pos], y=[vce_on], mode='markers',
                                  name=f'Current State (Vce={vce_on}V)',
                                  marker=dict(size=14, color=health_to_color(component_healths["IGBT Module"]),
                                              line=dict(width=2, color='white'))))

    fig_rul.update_layout(
        xaxis_title="Aging Time (hours)",
        yaxis_title="Vce_on (V)",
        height=320,
        margin=dict(t=10, b=50, l=50, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
    )
    st.plotly_chart(fig_rul, use_container_width=True)

st.markdown("---")

# ── ROW 5: KG Visualization ──
st.markdown("### 🕸️ Knowledge Graph — Fault Propagation Network")
st.markdown("*Interactive view of the causal chain from component degradation to community impact.*")

# Build graph using plotly (networkx for layout)
import networkx as nx

G = nx.DiGraph()

# Add nodes by layer
layers = {
    0: [("IGBT", "#3182ce"), ("Capacitor", "#3182ce"), ("Bearings", "#3182ce"), ("Seal", "#3182ce")],
    1: [("Bond Wire\nLift-off", "#805ad5"), ("ESR\nDrift", "#805ad5"), ("Bearing\nWear", "#805ad5"), ("Seal\nDeterioration", "#805ad5")],
    2: [("Vce_on\nRise", "#d69e2e"), ("DC Bus\nRipple", "#d69e2e"), ("Vibration\nIncrease", "#d69e2e"), ("Seal\nLeakage", "#d69e2e")],
    3: [("Conduction\nLoss ↑", "#dd6b20"), ("Harmonic\nDistortion", "#dd6b20"), ("Thermal\nResistance ↑", "#dd6b20")],
    4: [("VFD\nTrip", "#e53e3e"), ("Pump\nSeizure", "#e53e3e"), ("Motor\nFailure", "#e53e3e")],
    5: [("Station\nOffline", "#1a1a2e"), ("Boil Water\nAdvisory", "#1a1a2e"), ("Emergency\nDistribution", "#1a1a2e")],
}

# Position nodes in layers
pos = {}
for layer, nodes in layers.items():
    n = len(nodes)
    for i, (name, color) in enumerate(nodes):
        x = layer * 1.8
        y = (i - (n - 1) / 2) * 1.5
        pos[name] = (x, y)
        G.add_node(name, color=color, layer=layer)

# Add edges
edges = [
    ("IGBT", "Bond Wire\nLift-off"), ("Capacitor", "ESR\nDrift"),
    ("Bearings", "Bearing\nWear"), ("Seal", "Seal\nDeterioration"),
    ("Bond Wire\nLift-off", "Vce_on\nRise"), ("ESR\nDrift", "DC Bus\nRipple"),
    ("Bearing\nWear", "Vibration\nIncrease"), ("Seal\nDeterioration", "Seal\nLeakage"),
    ("Vce_on\nRise", "Conduction\nLoss ↑"), ("Vce_on\nRise", "Thermal\nResistance ↑"),
    ("DC Bus\nRipple", "Harmonic\nDistortion"),
    ("Conduction\nLoss ↑", "VFD\nTrip"), ("Thermal\nResistance ↑", "VFD\nTrip"),
    ("Harmonic\nDistortion", "VFD\nTrip"),
    ("Vibration\nIncrease", "Pump\nSeizure"), ("Seal\nLeakage", "Motor\nFailure"),
    ("VFD\nTrip", "Station\nOffline"), ("Pump\nSeizure", "Station\nOffline"),
    ("Pump\nSeizure", "Boil Water\nAdvisory"), ("Motor\nFailure", "Boil Water\nAdvisory"),
    ("Motor\nFailure", "Emergency\nDistribution"),
]
G.add_edges_from(edges)

# Create plotly figure
edge_x, edge_y = [], []
for e in G.edges():
    x0, y0 = pos[e[0]]
    x1, y1 = pos[e[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig_kg = go.Figure()

fig_kg.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                             line=dict(width=1.5, color='#cbd5e0'),
                             hoverinfo='none'))

# Color nodes based on health status
node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

    layer = G.nodes[node].get("layer", 0)
    base_color = G.nodes[node].get("color", "#666")

    # Highlight active fault paths
    if layer == 0:
        health_map = {"IGBT": component_healths["IGBT Module"],
                      "Capacitor": component_healths["DC-Link Capacitor"],
                      "Bearings": component_healths["Pump Bearings"],
                      "Seal": component_healths["Mechanical Seal"]}
        h = health_map.get(node, 100)
        node_color.append(health_to_color(h))
        node_size.append(30)
    else:
        node_color.append(base_color)
        node_size.append(22 if layer < 4 else 28)

fig_kg.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             marker=dict(size=node_size, color=node_color,
                                          line=dict(width=2, color='white')),
                             text=node_text, textposition="bottom center",
                             textfont=dict(size=9, family="DM Sans"),
                             hoverinfo='text'))

# Layer labels
layer_labels = ["Components", "Degradation\nModes", "Indicators", "Subsystem\nEffects",
                "System\nFaults", "Service\nImpacts"]
for i, label in enumerate(layer_labels):
    fig_kg.add_annotation(x=i * 1.8, y=3.5, text=f"<b>{label}</b>",
                           showarrow=False, font=dict(size=11, color="#4a5568"))

fig_kg.update_layout(
    showlegend=False,
    height=450,
    margin=dict(t=20, b=20, l=20, r=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
)
st.plotly_chart(fig_kg, use_container_width=True)

# ── ROW 6: Preventive vs Reactive Cost ──
st.markdown("---")
st.markdown("### 💡 Maintenance Decision: Preventive vs. Reactive")

col_prev, col_react = st.columns(2)

preventive_cost = sum(
    {"IGBT Module": 2500, "DC-Link Capacitor": 800, "Pump Bearings": 1200,
     "Mechanical Seal": 900, "Gate Driver": 350, "Stator Winding": 5000,
     "Impeller": 600, "Diode Bridge Rectifier": 200}.get(name, 500)
    for name, h in component_healths.items() if h < 50
)
reactive_cost = preventive_cost * 3.5 + reg_penalty if system_risk > 25 else preventive_cost

with col_prev:
    st.markdown(f"""
    <div style="background:#f0fff4; border-radius:12px; padding:20px; border:2px solid #c6f6d5; text-align:center;">
        <div style="font-size:0.85rem; color:#38a169; text-transform:uppercase; letter-spacing:1px;">Preventive Maintenance (Now)</div>
        <div style="font-size:2.8rem; font-weight:700; font-family:'JetBrains Mono'; color:#2f855a;">${preventive_cost:,.0f}</div>
        <div style="font-size:0.9rem; color:#666;">Planned downtime: 8-16 hours<br/>Zero regulatory risk<br/>Zero population impact</div>
    </div>
    """, unsafe_allow_html=True)

with col_react:
    st.markdown(f"""
    <div style="background:#fff5f5; border-radius:12px; padding:20px; border:2px solid #fed7d7; text-align:center;">
        <div style="font-size:0.85rem; color:#e53e3e; text-transform:uppercase; letter-spacing:1px;">Reactive Repair (After Failure)</div>
        <div style="font-size:2.8rem; font-weight:700; font-family:'JetBrains Mono'; color:#c53030;">${reactive_cost:,.0f}</div>
        <div style="font-size:0.9rem; color:#666;">Emergency downtime: {downtime}-72 hours<br/>Regulatory penalties: ${reg_penalty:,.0f}<br/>Population affected: {pop_at_risk:,}</div>
    </div>
    """, unsafe_allow_html=True)

if preventive_cost > 0 and reactive_cost > preventive_cost:
    savings = reactive_cost - preventive_cost
    roi = (savings / preventive_cost) * 100
    st.markdown(f"""
    <div style="background:#ebf8ff; border-radius:12px; padding:16px; margin-top:12px; text-align:center; border:1px solid #bee3f8;">
        <span style="font-size:1.1rem; color:#2b6cb0; font-weight:600;">
            Preventive maintenance saves ${savings:,.0f} ({roi:.0f}% ROI) and protects {pop_at_risk:,} residents
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.8rem;">
    Water Pump Infrastructure Risk Analytics | Knowledge Graph-Driven Prognostics Framework<br/>
    Department of Electrical and Computer Engineering | Mississippi State University | 2026
</div>
""", unsafe_allow_html=True)
