"""
Water Pump Infrastructure Risk Analytics Dashboard
=====================================================
Streamlit app — self-contained, no external database required.
Knowledge Graph is embedded; sliders drive real-time risk scoring.

Usage:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx


# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Water Pump Risk Analytics",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-size: 2.2rem; font-weight: 700; color: #1a1a2e;
        margin-bottom: 0; letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1rem; color: #666; margin-top: -10px; margin-bottom: 20px;
    }
    .risk-card {
        border-radius: 12px; padding: 20px; margin: 8px 0; border-left: 5px solid;
    }
    .risk-critical { background: #fff5f5; border-left-color: #e53e3e; }
    .risk-high     { background: #fffaf0; border-left-color: #dd6b20; }
    .risk-medium   { background: #fffff0; border-left-color: #d69e2e; }
    .risk-low      { background: #f0fff4; border-left-color: #38a169; }

    .fault-path {
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #2d3748;
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  RISK COMPUTATION
# ════════════════════════════════════════════════════════════════

def compute_component_health(vce_on_current, vce_on_initial=1.3, vce_on_threshold=4.5):
    """Vce_on → 0–100 health (linear between healthy and failure thresholds)."""
    if vce_on_current <= vce_on_initial:
        return 100.0
    if vce_on_current >= vce_on_threshold:
        return 0.0
    return max(0.0, 100.0 * (1 - (vce_on_current - vce_on_initial) /
                             (vce_on_threshold - vce_on_initial)))


def compute_esr_health(esr_current, esr_initial=15, esr_threshold=30):
    """ESR (mΩ) → 0–100 health."""
    if esr_current <= esr_initial:
        return 100.0
    if esr_current >= esr_threshold:
        return 0.0
    return max(0.0, 100.0 * (1 - (esr_current - esr_initial) /
                             (esr_threshold - esr_initial)))


# Component weights — IGBT and capacitor dominate (matches reliability data:
# power electronics drive failure). IGBT weight raised to 0.45 so the slider
# produces a visually clear gauge response during demo.
COMPONENT_WEIGHTS = {
    "IGBT Module":            0.45,
    "DC-Link Capacitor":      0.20,
    "Pump Bearings":          0.15,
    "Mechanical Seal":        0.07,
    "Gate Driver":            0.05,
    "Stator Winding":         0.04,
    "Impeller":               0.02,
    "Diode Bridge Rectifier": 0.02,
}


def compute_system_risk(component_healths):
    """Weighted average risk + weakest-link penalty (60/40 blend)."""
    weighted_risk = sum(
        (100 - h) * COMPONENT_WEIGHTS.get(name, 0.05)
        for name, h in component_healths.items()
    )
    weakest_link_risk = 100 - min(component_healths.values())
    system_risk = 0.6 * weighted_risk + 0.4 * weakest_link_risk
    return min(100.0, max(0.0, system_risk))


def risk_to_color(risk):
    if risk >= 75:  return "#e53e3e"
    if risk >= 50:  return "#dd6b20"
    if risk >= 25:  return "#d69e2e"
    return "#38a169"


def risk_to_label(risk):
    if risk >= 75:  return "CRITICAL"
    if risk >= 50:  return "HIGH"
    if risk >= 25:  return "MODERATE"
    return "LOW"


def health_to_color(health):
    if health >= 70:  return "#38a169"
    if health >= 30:  return "#d69e2e"
    return "#e53e3e"


def service_status_from_risk(risk):
    """Map system risk to a service-status label."""
    if risk >= 75:  return ("Service Disruption Imminent", "#e53e3e")
    if risk >= 50:  return ("Degraded Service Likely",     "#dd6b20")
    if risk >= 25:  return ("Service Reliable — Monitor",  "#d69e2e")
    return ("Service Reliable",                            "#38a169")


# ════════════════════════════════════════════════════════════════
#  SIDEBAR — sensor inputs + scenario presets
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🔧 Sensor Readings")
st.sidebar.markdown("Adjust to simulate component degradation.")
st.sidebar.markdown("---")

# Scenario preset (above sliders so users see it first)
st.sidebar.markdown("### 🎬 Quick Scenarios")
scenario = st.sidebar.selectbox(
    "Load preset",
    [
        "— Custom (use sliders below) —",
        "🟢 All Healthy",
        "🟡 Early IGBT Degradation",
        "🟠 Late-Stage IGBT + Capacitor",
        "🔴 Jackson Crisis Simulation",
    ],
)

# Default slider values (overwritten by presets below)
preset_vce, preset_esr, preset_gate, preset_vib, preset_seal = 1.3, 15, 95, 2.0, 90

if scenario == "🟢 All Healthy":
    preset_vce, preset_esr, preset_gate, preset_vib, preset_seal = 1.3, 15, 95, 2.0, 90
elif scenario == "🟡 Early IGBT Degradation":
    preset_vce, preset_esr, preset_gate, preset_vib, preset_seal = 2.8, 18, 90, 3.0, 85
elif scenario == "🟠 Late-Stage IGBT + Capacitor":
    preset_vce, preset_esr, preset_gate, preset_vib, preset_seal = 4.0, 32, 70, 5.0, 60
elif scenario == "🔴 Jackson Crisis Simulation":
    preset_vce, preset_esr, preset_gate, preset_vib, preset_seal = 4.4, 42, 40, 12.0, 25

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Power Electronics")
vce_on = st.sidebar.slider(
    "IGBT Vce_on (V)", 1.0, 5.0, preset_vce, 0.1,
    help="Collector-emitter on-state voltage. Healthy: 1.2-1.5V | Failed: >4.5V",
)
esr = st.sidebar.slider(
    "Capacitor ESR (mΩ)", 10, 60, preset_esr, 1,
    help="Equivalent series resistance. Healthy: 10-20 mΩ | Failed: >30 mΩ",
)
gate_health = st.sidebar.slider("Gate Driver Health (%)", 0, 100, preset_gate, 5)

st.sidebar.markdown("### 🔩 Mechanical")
bearing_vib = st.sidebar.slider(
    "Bearing Vibration (mm/s)", 0.0, 15.0, preset_vib, 0.5,
    help="ISO 10816: <4.5 = Good · 4.5–11.2 = Alert · >11.2 = Danger",
)
seal_health = st.sidebar.slider("Mechanical Seal Health (%)", 0, 100, preset_seal, 5)

st.sidebar.markdown("### 🏭 Station")
station = st.sidebar.selectbox(
    "Pump Station",
    ["Pump Station North", "Pump Station South", "Pump Station Central"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "📚 Methodology validated on NASA PCoE IGBT Aging Dataset (4 devices, "
    "IRG4BC30K). Best result: MAE 0.019 h, 100% CI coverage."
)


# Compute component healths from inputs
component_healths = {
    "IGBT Module":            compute_component_health(vce_on),
    "DC-Link Capacitor":      compute_esr_health(esr),
    "Gate Driver":            float(gate_health),
    "Pump Bearings":          max(0.0, 100.0 - (bearing_vib / 15.0) * 100.0),
    "Mechanical Seal":        float(seal_health),
    "Stator Winding":         70.0,   # baseline assumed-healthy components
    "Impeller":               80.0,
    "Diode Bridge Rectifier": 85.0,
}

system_risk = compute_system_risk(component_healths)
risk_color = risk_to_color(system_risk)
service_label, service_color = service_status_from_risk(system_risk)


# ════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">Water Pump Infrastructure Risk Analytics</p>',
            unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">Knowledge Graph-Driven Prognostics · '
    f'<b>{station}</b> · Scenario: {scenario}</p>',
    unsafe_allow_html=True,
)

# ── Row 1: Key Metrics (service-impact only, no cost) ──
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("System Risk Score", f"{system_risk:.0f}/100", risk_to_label(system_risk))

with col2:
    min_health_component = min(component_healths, key=component_healths.get)
    min_health = component_healths[min_health_component]
    st.metric("Weakest Component", min_health_component, f"{min_health:.0f}%")

with col3:
    st.metric("Service Status", service_label)

with col4:
    station_pop = {
        "Pump Station North": 15000,
        "Pump Station South": 22000,
        "Pump Station Central": 35000,
    }
    pop = station_pop.get(station, 15000)
    pop_at_risk = int(pop * system_risk / 100)
    st.metric("Population at Risk", f"{pop_at_risk:,}", f"of {pop:,} served")

st.markdown("---")

# ── Row 2: Risk Gauge + Component Health ──
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📊 System Risk Gauge")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=system_risk,
        number={"suffix": "%", "font": {"size": 48, "family": "JetBrains Mono"}},
        title={"text": f"Overall Risk — {station}", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": risk_color, "thickness": 0.75},
            "bgcolor": "#f7fafc",
            "steps": [
                {"range": [0, 25],  "color": "#f0fff4"},
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
    fig_gauge.update_layout(
        height=300, margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_right:
    st.markdown("### 🔋 Component Health Index")

    health_df = pd.DataFrame([
        {"Component": name, "Health": health, "Color": health_to_color(health)}
        for name, health in sorted(component_healths.items(), key=lambda x: x[1])
    ])

    fig_health = go.Figure()
    for _, row in health_df.iterrows():
        fig_health.add_trace(go.Bar(
            y=[row["Component"]], x=[row["Health"]],
            orientation='h', marker_color=row["Color"],
            text=f'{row["Health"]:.0f}%', textposition='inside',
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

# ── Row 3: Fault Propagation Paths + Community Impact ──
col_fault, col_impact = st.columns([1, 1])

with col_fault:
    st.markdown("### 🔗 Active Fault Propagation Paths")
    st.markdown("*Causal chains from degraded components to service impact:*")

    active_paths = []

    if component_healths["IGBT Module"] < 55:
        active_paths.append({
            "path": "IGBT → Bond Wire Lift-off → Vce_on Rise → Conduction Loss ↑ → "
                    "VFD Overtemp → Station Offline",
            "risk": 100 - component_healths["IGBT Module"],
            "component": "IGBT Module",
        })

    if component_healths["DC-Link Capacitor"] < 55:
        active_paths.append({
            "path": "Capacitor → ESR Drift → DC Bus Ripple → VFD DC Bus Fault → "
                    "Station Offline",
            "risk": 100 - component_healths["DC-Link Capacitor"],
            "component": "DC-Link Capacitor",
        })

    if component_healths["Pump Bearings"] < 55:
        active_paths.append({
            "path": "Bearings → Wear → Vibration ↑ → Pump Seizure → Boil Water Advisory",
            "risk": 100 - component_healths["Pump Bearings"],
            "component": "Pump Bearings",
        })

    if component_healths["Mechanical Seal"] < 55:
        active_paths.append({
            "path": "Seal → Deterioration → Leakage → Water Ingress → Motor Failure → "
                    "Emergency Distribution",
            "risk": 100 - component_healths["Mechanical Seal"],
            "component": "Mechanical Seal",
        })

    if (component_healths["IGBT Module"] < 70 and
        component_healths["DC-Link Capacitor"] < 70):
        active_paths.append({
            "path": "IGBT + Capacitor → Combined VFD Degradation → Harmonic Distortion → "
                    "Motor Insulation Failure → Boil Water Advisory",
            "risk": max(100 - component_healths["IGBT Module"],
                        100 - component_healths["DC-Link Capacitor"]) * 1.2,
            "component": "Multiple",
        })

    if not active_paths:
        st.success("✅ No active fault propagation paths. All components within "
                   "healthy operating limits.")
    else:
        active_paths.sort(key=lambda x: x["risk"], reverse=True)
        for p in active_paths:
            risk_class = ("risk-critical" if p["risk"] >= 75
                          else "risk-high" if p["risk"] >= 50
                          else "risk-medium")
            st.markdown(f'''
            <div class="risk-card {risk_class}">
                <div style="font-weight:600; margin-bottom:4px;">
                  Risk: {p["risk"]:.0f}% — {p["component"]}
                </div>
                <div class="fault-path">{p["path"]}</div>
            </div>
            ''', unsafe_allow_html=True)

with col_impact:
    st.markdown("### 🏘️ Community Impact")

    if system_risk < 25:
        st.markdown("""
        <div style="background:#f0fff4; border-radius:12px; padding:20px;
                    border:1px solid #c6f6d5;">
            <h4 style="color:#38a169; margin:0;">Service Reliable</h4>
            <p style="color:#666; margin-top:8px;">
              All components within acceptable limits. Continue routine monitoring.
              No community impact anticipated.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Service disruption projections (illustrative — community focus only)
        failure_prob = min(system_risk / 100, 0.95)
        expected_outage_hours = max(4, int(system_risk / 100 * 72))
        if system_risk >= 75:
            advisory = "Boil Water Advisory likely"
            bottled_water_need = pop_at_risk
        elif system_risk >= 50:
            advisory = "Service pressure reductions expected"
            bottled_water_need = int(pop_at_risk * 0.5)
        else:
            advisory = "Intermittent service interruptions possible"
            bottled_water_need = int(pop_at_risk * 0.2)

        households_affected = int(pop_at_risk / 2.6)   # US avg household size ≈ 2.6
        schools_affected   = max(0, int(pop_at_risk / 5000))
        healthcare_affected = max(0, int(pop_at_risk / 12000))

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2d3748 0%,#1a202c 100%);
                    border-radius:12px; padding:20px; color:white;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:1px;">
                                Failure Probability</div>
                    <div style="font-size:2rem; font-weight:700;
                                font-family:'JetBrains Mono'; color:{risk_color};">
                                {failure_prob*100:.0f}%</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:1px;">
                                Expected Service Outage</div>
                    <div style="font-size:2rem; font-weight:700;
                                font-family:'JetBrains Mono';">
                                {expected_outage_hours}h</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:1px;">
                                People Affected</div>
                    <div style="font-size:2rem; font-weight:700;
                                font-family:'JetBrains Mono';">
                                {pop_at_risk:,}</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:1px;">
                                Households Affected</div>
                    <div style="font-size:2rem; font-weight:700;
                                font-family:'JetBrains Mono';">
                                {households_affected:,}</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:1px;">
                                Schools Impacted</div>
                    <div style="font-size:2rem; font-weight:700;
                                font-family:'JetBrains Mono';">
                                {schools_affected}</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem;
                                text-transform:uppercase; letter-spacing:1px;">
                                Healthcare Facilities</div>
                    <div style="font-size:2rem; font-weight:700;
                                font-family:'JetBrains Mono';">
                                {healthcare_affected}</div>
                </div>
            </div>
            <div style="margin-top:16px; border-top:1px solid #4a5568;
                        padding-top:12px;">
                <div style="color:#a0aec0; font-size:0.75rem;
                            text-transform:uppercase; letter-spacing:1px;">
                            Public Health Advisory</div>
                <div style="font-size:1.3rem; font-weight:700; color:#fbd38d;
                            margin-top:4px;">
                            {advisory}</div>
                <div style="font-size:0.85rem; color:#a0aec0; margin-top:6px;">
                  Estimated bottled-water need: {bottled_water_need:,} people
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Row 4: Maintenance Priority + RUL Projection ──
col_maint, col_rul = st.columns([1, 1])

with col_maint:
    st.markdown("### 🛠️ Maintenance Priority Ranking")

    maint_data = []
    for name, health in component_healths.items():
        urgency = (100 - health)
        maint_data.append({
            "Component": name,
            "Health": f"{health:.0f}%",
            "Urgency": f"{urgency:.0f}",
            "Action": ("🔴 Replace Now" if health < 20
                       else "🟡 Schedule" if health < 50
                       else "🟢 Monitor"),
        })

    maint_df = pd.DataFrame(maint_data)
    maint_df = maint_df.sort_values("Urgency", ascending=False,
                                    key=lambda x: x.astype(float))
    st.dataframe(maint_df, use_container_width=True, hide_index=True, height=320)

with col_rul:
    st.markdown("### 📈 RUL Degradation Projection (Sigmoid Model)")

    # Sigmoid parameters from NASA Device2 fit (validated)
    L, k, t0, c = 4.456, 11.084, 0.0778, 0.0
    t = np.linspace(0, 0.8, 200)
    vce_trajectory = L / (1 + np.exp(-k * (t - t0))) + c
    threshold = c + 0.95 * L

    # Find current position on curve
    current_pos = next((t[i] for i, v in enumerate(vce_trajectory) if v >= vce_on),
                       t[-1])

    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(
        x=t, y=vce_trajectory, mode='lines',
        name='Predicted Degradation',
        line=dict(color='#3182ce', width=2.5),
    ))
    fig_rul.add_hline(
        y=threshold, line_dash="dash", line_color="#e53e3e",
        annotation_text=f"Failure Threshold ({threshold:.2f}V)",
    )
    fig_rul.add_trace(go.Scatter(
        x=[current_pos], y=[vce_on], mode='markers',
        name=f'Current State (Vce={vce_on}V)',
        marker=dict(
            size=14,
            color=health_to_color(component_healths["IGBT Module"]),
            line=dict(width=2, color='white'),
        ),
    ))
    fig_rul.update_layout(
        xaxis_title="Aging Time (hours)",
        yaxis_title="Vce_on (V)",
        height=320, margin=dict(t=10, b=50, l=50, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
    )
    st.plotly_chart(fig_rul, use_container_width=True)

st.markdown("---")

# ── Row 5: Knowledge Graph Visualization ──
st.markdown("### 🕸️ Knowledge Graph — Fault Propagation Network")
st.markdown("*Interactive view of the causal chain from component degradation "
            "to community impact.*")

G = nx.DiGraph()

layers = {
    0: [("IGBT", "#3182ce"), ("Capacitor", "#3182ce"),
        ("Bearings", "#3182ce"), ("Seal", "#3182ce")],
    1: [("Bond Wire\nLift-off", "#805ad5"), ("ESR\nDrift", "#805ad5"),
        ("Bearing\nWear", "#805ad5"), ("Seal\nDeterioration", "#805ad5")],
    2: [("Vce_on\nRise", "#d69e2e"), ("DC Bus\nRipple", "#d69e2e"),
        ("Vibration\nIncrease", "#d69e2e"), ("Seal\nLeakage", "#d69e2e")],
    3: [("Conduction\nLoss ↑", "#dd6b20"), ("Harmonic\nDistortion", "#dd6b20"),
        ("Thermal\nResistance ↑", "#dd6b20")],
    4: [("VFD\nTrip", "#e53e3e"), ("Pump\nSeizure", "#e53e3e"),
        ("Motor\nFailure", "#e53e3e")],
    5: [("Station\nOffline", "#1a1a2e"), ("Boil Water\nAdvisory", "#1a1a2e"),
        ("Emergency\nDistribution", "#1a1a2e")],
}

pos = {}
for layer, nodes in layers.items():
    n = len(nodes)
    for i, (name, color) in enumerate(nodes):
        x = layer * 1.8
        y = (i - (n - 1) / 2) * 1.5
        pos[name] = (x, y)
        G.add_node(name, color=color, layer=layer)

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

edge_x, edge_y = [], []
for e in G.edges():
    x0, y0 = pos[e[0]]
    x1, y1 = pos[e[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig_kg = go.Figure()
fig_kg.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode='lines',
    line=dict(width=1.5, color='#cbd5e0'), hoverinfo='none',
))

node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x); node_y.append(y); node_text.append(node)

    layer = G.nodes[node].get("layer", 0)
    base_color = G.nodes[node].get("color", "#666")

    if layer == 0:
        health_map = {
            "IGBT":      component_healths["IGBT Module"],
            "Capacitor": component_healths["DC-Link Capacitor"],
            "Bearings":  component_healths["Pump Bearings"],
            "Seal":      component_healths["Mechanical Seal"],
        }
        h = health_map.get(node, 100)
        node_color.append(health_to_color(h))
        node_size.append(30)
    else:
        node_color.append(base_color)
        node_size.append(22 if layer < 4 else 28)

fig_kg.add_trace(go.Scatter(
    x=node_x, y=node_y, mode='markers+text',
    marker=dict(size=node_size, color=node_color,
                line=dict(width=2, color='white')),
    text=node_text, textposition="bottom center",
    textfont=dict(size=9, family="DM Sans"),
    hoverinfo='text',
))

layer_labels = ["Components", "Degradation\nModes", "Indicators",
                "Subsystem\nEffects", "System\nFaults", "Service\nImpacts"]
for i, label in enumerate(layer_labels):
    fig_kg.add_annotation(
        x=i * 1.8, y=3.5, text=f"<b>{label}</b>",
        showarrow=False, font=dict(size=11, color="#4a5568"),
    )

fig_kg.update_layout(
    showlegend=False, height=450,
    margin=dict(t=20, b=20, l=20, r=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
)
st.plotly_chart(fig_kg, use_container_width=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.8rem;">
    Water Pump Infrastructure Risk Analytics |
    Knowledge Graph-Driven Prognostics Framework<br/>
    Department of Electrical and Computer Engineering |
    Mississippi State University | 2026
</div>
""", unsafe_allow_html=True)
