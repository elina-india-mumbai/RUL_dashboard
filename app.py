"""
Water Pump Infrastructure Risk Analytics Dashboard
=====================================================
Self-contained Streamlit app with embedded Knowledge Graph.
No external database required — deployable on Streamlit Cloud.

GitHub → Streamlit Cloud deployment:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io
    3. Connect your GitHub repo
    4. Set app.py as the main file

Local run:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx

# ══════════════════════════════════════════════════════════════
#  EMBEDDED KNOWLEDGE GRAPH DATA
# ══════════════════════════════════════════════════════════════

COMPONENTS = {
    "IGBT Module": {
        "category": "Power Electronics", "criticality": "Critical",
        "mtbf_hours": 50000, "replacement_cost": 2500, "lead_time_days": 14,
        "description": "Main switching device in VFD inverter stage",
        "icon": "⚡", "weight": 0.30,
    },
    "DC-Link Capacitor": {
        "category": "Power Electronics", "criticality": "Critical",
        "mtbf_hours": 40000, "replacement_cost": 800, "lead_time_days": 7,
        "description": "DC bus energy storage and voltage smoothing",
        "icon": "🔋", "weight": 0.25,
    },
    "Gate Driver": {
        "category": "Power Electronics", "criticality": "High",
        "mtbf_hours": 80000, "replacement_cost": 350, "lead_time_days": 5,
        "description": "Switching signal circuit for IGBT",
        "icon": "📡", "weight": 0.05,
    },
    "Pump Bearings": {
        "category": "Mechanical", "criticality": "High",
        "mtbf_hours": 30000, "replacement_cost": 1200, "lead_time_days": 10,
        "description": "Radial and thrust bearings supporting pump shaft",
        "icon": "🔩", "weight": 0.20,
    },
    "Mechanical Seal": {
        "category": "Mechanical", "criticality": "High",
        "mtbf_hours": 25000, "replacement_cost": 900, "lead_time_days": 7,
        "description": "Shaft seal preventing water ingress",
        "icon": "🛡️", "weight": 0.10,
    },
    "Stator Winding": {
        "category": "Electrical", "criticality": "Critical",
        "mtbf_hours": 70000, "replacement_cost": 5000, "lead_time_days": 21,
        "description": "Motor stator — converts electrical to mechanical energy",
        "icon": "🌀", "weight": 0.05,
    },
    "Impeller": {
        "category": "Mechanical", "criticality": "Medium",
        "mtbf_hours": 60000, "replacement_cost": 600, "lead_time_days": 14,
        "description": "Rotating element moving water through pump",
        "icon": "💧", "weight": 0.03,
    },
    "Diode Bridge Rectifier": {
        "category": "Power Electronics", "criticality": "Medium",
        "mtbf_hours": 100000, "replacement_cost": 200, "lead_time_days": 3,
        "description": "AC-DC rectifier stage of VFD",
        "icon": "🔌", "weight": 0.02,
    },
}

FAULT_CHAINS = [
    {
        "id": "igbt_bondwire",
        "trigger_component": "IGBT Module",
        "trigger_threshold": 50,
        "path": ["IGBT Module", "Bond Wire Lift-off", "Vce_on Rise",
                 "Increased Conduction Loss", "VFD Overtemperature Shutdown", "Pump Station Offline"],
        "path_display": "IGBT → Bond Wire Lift-off → Vce_on Rise → Conduction Loss ↑ → VFD Overtemp → Station Offline",
        "end_impact": "Pump Station Offline",
        "physics": "CTE mismatch causes fatigue at wire-chip interface. Rising Vce_on increases I²R losses, generating excess heat until thermal protection trips.",
    },
    {
        "id": "igbt_solder",
        "trigger_component": "IGBT Module",
        "trigger_threshold": 40,
        "path": ["IGBT Module", "Solder Layer Fatigue", "Thermal Resistance Rise",
                 "VFD Overcurrent Trip", "Pump Station Offline"],
        "path_display": "IGBT → Solder Fatigue → Thermal Resistance ↑ → VFD Overcurrent Trip → Station Offline",
        "end_impact": "Pump Station Offline",
        "physics": "Solder crack propagation reduces die-to-heatsink heat transfer. Junction temperature rises until overcurrent protection activates.",
    },
    {
        "id": "cap_esr",
        "trigger_component": "DC-Link Capacitor",
        "trigger_threshold": 50,
        "path": ["DC-Link Capacitor", "ESR Drift", "DC Bus Voltage Ripple",
                 "VFD DC Bus Fault", "Pump Station Offline"],
        "path_display": "Capacitor → ESR Drift → DC Bus Ripple → VFD DC Bus Fault → Station Offline",
        "end_impact": "Pump Station Offline",
        "physics": "Electrolyte evaporation increases ESR. Higher ripple causes bus overvoltage/undervoltage protection to activate.",
    },
    {
        "id": "cap_harmonic",
        "trigger_component": "DC-Link Capacitor",
        "trigger_threshold": 40,
        "path": ["DC-Link Capacitor", "Capacitance Fade", "Output Harmonic Distortion",
                 "Motor Insulation Failure", "Boil Water Advisory"],
        "path_display": "Capacitor → Capacitance Fade → Harmonic Distortion → Motor Insulation Failure → Boil Water Advisory",
        "end_impact": "Boil Water Advisory",
        "physics": "Reduced capacitance degrades DC bus filtering. Harmonic-rich motor current causes partial discharge erosion of stator insulation.",
    },
    {
        "id": "bearing_seizure",
        "trigger_component": "Pump Bearings",
        "trigger_threshold": 50,
        "path": ["Pump Bearings", "Bearing Wear", "Vibration Increase",
                 "Pump Seizure", "Boil Water Advisory"],
        "path_display": "Bearings → Wear → Vibration ↑ → Pump Seizure → Boil Water Advisory",
        "end_impact": "Boil Water Advisory",
        "physics": "Surface fatigue and spalling degrade bearing raceways. Increasing vibration leads to shaft lockup and motor overcurrent trip.",
    },
    {
        "id": "seal_ingress",
        "trigger_component": "Mechanical Seal",
        "trigger_threshold": 40,
        "path": ["Mechanical Seal", "Seal Deterioration", "Seal Leakage",
                 "Water Ingress to Motor", "Emergency Water Distribution"],
        "path_display": "Seal → Deterioration → Leakage → Water Ingress → Motor Failure → Emergency Distribution",
        "end_impact": "Emergency Water Distribution",
        "physics": "Elastomer aging and face wear allow water past the seal into the motor cavity, causing winding insulation failure.",
    },
    {
        "id": "combined_vfd",
        "trigger_component": "IGBT Module",
        "trigger_threshold": 70,
        "secondary_component": "DC-Link Capacitor",
        "secondary_threshold": 70,
        "path": ["IGBT Module + Capacitor", "Combined VFD Degradation",
                 "Harmonic Distortion + Thermal Stress", "Motor Insulation Failure", "Boil Water Advisory"],
        "path_display": "IGBT + Capacitor → Combined VFD Degradation → Harmonics + Thermal Stress → Motor Failure → Boil Water Advisory",
        "end_impact": "Boil Water Advisory",
        "physics": "Simultaneous IGBT and capacitor degradation compounds VFD output quality issues, accelerating motor winding failure.",
    },
]

PUMP_STATIONS = {
    "Pump Station North": {"location": "North Zone", "capacity_mgd": 4.5, "age_years": 18, "redundancy": "N+1", "population_served": 15000},
    "Pump Station South": {"location": "South Zone", "capacity_mgd": 6.0, "age_years": 25, "redundancy": "None", "population_served": 22000},
    "Pump Station Central": {"location": "Central Zone", "capacity_mgd": 8.0, "age_years": 12, "redundancy": "N+1", "population_served": 35000},
}

IMPACT_DATA = {
    "Pump Station Offline": {"population_factor": 0.4, "duration_hours": 8, "direct_cost": 15000, "regulatory_penalty": 5000, "category": "Infrastructure"},
    "Reduced Water Pressure": {"population_factor": 0.25, "duration_hours": 4, "direct_cost": 5000, "regulatory_penalty": 2000, "category": "Service Quality"},
    "Boil Water Advisory": {"population_factor": 0.8, "duration_hours": 72, "direct_cost": 50000, "regulatory_penalty": 100000, "category": "Public Health"},
    "Emergency Water Distribution": {"population_factor": 0.5, "duration_hours": 48, "direct_cost": 75000, "regulatory_penalty": 0, "category": "Emergency Response"},
    "Treatment Bypass": {"population_factor": 1.0, "duration_hours": 24, "direct_cost": 200000, "regulatory_penalty": 500000, "category": "Environmental"},
}

SIGMOID_PARAMS = {"L": 4.456, "k": 11.084, "t0": 0.0778, "c": 0.0}


# ══════════════════════════════════════════════════════════════
#  COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def compute_health_vce(vce_on, initial=1.3, threshold=4.5):
    if vce_on <= initial: return 100.0
    if vce_on >= threshold: return 0.0
    return max(0, 100.0 * (1 - (vce_on - initial) / (threshold - initial)))

def compute_health_esr(esr, initial=15, threshold=30):
    if esr <= initial: return 100.0
    if esr >= threshold: return 0.0
    return max(0, 100.0 * (1 - (esr - initial) / (threshold - initial)))

def compute_health_vibration(vib, threshold=15.0):
    return max(0, 100.0 * (1 - vib / threshold))

def compute_system_risk(healths):
    weighted = sum((100 - h) * COMPONENTS[n]["weight"] for n, h in healths.items())
    weakest = 100 - min(healths.values())
    return min(100, max(0, 0.6 * weighted + 0.4 * weakest))

def risk_color(r):
    if r >= 75: return "#e53e3e"
    if r >= 50: return "#dd6b20"
    if r >= 25: return "#d69e2e"
    return "#38a169"

def risk_label(r):
    if r >= 75: return "CRITICAL"
    if r >= 50: return "HIGH"
    if r >= 25: return "MODERATE"
    return "LOW"

def health_color(h):
    if h >= 70: return "#38a169"
    if h >= 30: return "#d69e2e"
    return "#e53e3e"

def get_active_chains(healths):
    active = []
    for chain in FAULT_CHAINS:
        comp = chain["trigger_component"]
        thresh = chain["trigger_threshold"]
        triggered = healths.get(comp, 100) < thresh
        if "secondary_component" in chain:
            triggered = healths.get(comp, 100) < chain["trigger_threshold"] and \
                        healths.get(chain["secondary_component"], 100) < chain["secondary_threshold"]
        if triggered:
            risk = 100 - healths.get(comp, 100)
            active.append({**chain, "risk": min(risk * 1.1, 100)})
    return sorted(active, key=lambda x: x["risk"], reverse=True)

def worst_impact(active_chains):
    if not active_chains:
        return None
    impact_severity = {"Pump Station Offline": 1, "Reduced Water Pressure": 0,
                       "Boil Water Advisory": 3, "Emergency Water Distribution": 2, "Treatment Bypass": 4}
    worst = max(active_chains, key=lambda x: impact_severity.get(x["end_impact"], 0))
    return worst["end_impact"]


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG + STYLES
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Water Pump Risk Analytics", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; letter-spacing: -0.5px; }
    .sub-header { font-size: 1rem; color: #718096; margin-top: -8px; margin-bottom: 24px; }
    .risk-card { border-radius: 12px; padding: 16px 20px; margin: 8px 0; border-left: 5px solid; }
    .risk-critical { background: #fff5f5; border-left-color: #e53e3e; }
    .risk-high { background: #fffaf0; border-left-color: #dd6b20; }
    .risk-medium { background: #fffff0; border-left-color: #d69e2e; }
    .risk-low { background: #f0fff4; border-left-color: #38a169; }
    .fault-path { background: #f7fafc; border-radius: 6px; padding: 8px 12px; margin: 4px 0;
                  border: 1px solid #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #2d3748; }
    .physics-note { font-size: 0.8rem; color: #718096; font-style: italic; margin-top: 4px; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    .cost-box { border-radius: 12px; padding: 24px; text-align: center; }
    .cost-value { font-size: 2.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .cost-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
    .cost-detail { font-size: 0.88rem; color: #666; line-height: 1.6; }
    .jackson-box { background: linear-gradient(135deg, #1a1a2e 0%, #2d3748 100%); border-radius: 12px;
                   padding: 24px; color: white; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🔧 Sensor Readings")
st.sidebar.markdown("Adjust to simulate component degradation.")
st.sidebar.markdown("---")

st.sidebar.markdown("#### ⚡ Power Electronics")
vce_on = st.sidebar.slider("IGBT Vce_on (V)", 1.0, 5.0, 1.3, 0.1, help="Healthy: 1.2–1.5V | Failed: >4.5V")
esr = st.sidebar.slider("Capacitor ESR (mΩ)", 10, 60, 15, 1, help="Healthy: 10–20 mΩ | Failed: >30 mΩ")
gate_health = st.sidebar.slider("Gate Driver Health (%)", 0, 100, 95, 5)

st.sidebar.markdown("#### 🔩 Mechanical")
bearing_vib = st.sidebar.slider("Bearing Vibration (mm/s)", 0.0, 15.0, 2.0, 0.5, help="ISO 10816: <4.5=Good | >11.2=Danger")
seal_health = st.sidebar.slider("Mechanical Seal Health (%)", 0, 100, 90, 5)

st.sidebar.markdown("#### 🏭 Station")
station = st.sidebar.selectbox("Pump Station", list(PUMP_STATIONS.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🎓 Quick Scenarios")
scenario = st.sidebar.selectbox("Load preset scenario", [
    "— Custom (use sliders above) —",
    "🟢 All Healthy",
    "🟡 Early IGBT Degradation",
    "🔴 Late-Stage IGBT + Capacitor",
    "🔴 Jackson Crisis Simulation",
])
if scenario == "🟢 All Healthy":
    vce_on, esr, gate_health, bearing_vib, seal_health = 1.3, 15, 95, 2.0, 90
elif scenario == "🟡 Early IGBT Degradation":
    vce_on, esr, gate_health, bearing_vib, seal_health = 2.8, 18, 90, 3.0, 85
elif scenario == "🔴 Late-Stage IGBT + Capacitor":
    vce_on, esr, gate_health, bearing_vib, seal_health = 4.2, 35, 70, 5.0, 60
elif scenario == "🔴 Jackson Crisis Simulation":
    vce_on, esr, gate_health, bearing_vib, seal_health = 4.5, 45, 40, 12.0, 25

# Compute health indices
healths = {
    "IGBT Module": compute_health_vce(vce_on),
    "DC-Link Capacitor": compute_health_esr(esr),
    "Gate Driver": float(gate_health),
    "Pump Bearings": compute_health_vibration(bearing_vib),
    "Mechanical Seal": float(seal_health),
    "Stator Winding": 85.0,
    "Impeller": 90.0,
    "Diode Bridge Rectifier": 95.0,
}
sys_risk = compute_system_risk(healths)
station_info = PUMP_STATIONS[station]
pop_served = station_info["population_served"]
active_chains = get_active_chains(healths)
worst = worst_impact(active_chains)


# ══════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════

st.markdown(f'<p class="main-header">Water Pump Infrastructure Risk Analytics</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Knowledge Graph-Driven Prognostics &nbsp;|&nbsp; {station} &nbsp;|&nbsp; {station_info["location"]} &nbsp;|&nbsp; {station_info["capacity_mgd"]} MGD &nbsp;|&nbsp; Age: {station_info["age_years"]} years</p>', unsafe_allow_html=True)

# ── Row 1: Key Metrics ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("System Risk Score", f"{sys_risk:.0f} / 100", risk_label(sys_risk),
              delta_color="inverse" if sys_risk > 50 else "normal")
with c2:
    weakest = min(healths, key=healths.get)
    st.metric("Weakest Component", f"{COMPONENTS[weakest]['icon']} {weakest}", f"Health: {healths[weakest]:.0f}%")
with c3:
    pop_at_risk = int(pop_served * sys_risk / 100)
    st.metric("Population at Risk", f"{pop_at_risk:,}", f"of {pop_served:,} served")
with c4:
    n_active = len(active_chains)
    st.metric("Active Fault Chains", f"{n_active}", "⚠️ Attention needed" if n_active > 0 else "✅ Clear")

st.markdown("---")

# ── Row 2: Risk Gauge + Component Health ──
col_gauge, col_health = st.columns([1, 1])

with col_gauge:
    st.markdown("#### 📊 System Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sys_risk,
        number={"suffix": "%", "font": {"size": 52, "family": "JetBrains Mono", "color": risk_color(sys_risk)}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "dtick": 25},
            "bar": {"color": risk_color(sys_risk), "thickness": 0.7},
            "bgcolor": "#edf2f7",
            "steps": [
                {"range": [0, 25], "color": "#f0fff4"},
                {"range": [25, 50], "color": "#fffff0"},
                {"range": [50, 75], "color": "#fffaf0"},
                {"range": [75, 100], "color": "#fff5f5"},
            ],
            "threshold": {"line": {"color": "#1a1a2e", "width": 3}, "thickness": 0.85, "value": sys_risk},
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=40, r=40), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_health:
    st.markdown("#### 🔋 Component Health Index")
    sorted_healths = sorted(healths.items(), key=lambda x: x[1])
    fig_bar = go.Figure()
    for name, h in sorted_healths:
        fig_bar.add_trace(go.Bar(
            y=[f"{COMPONENTS[name]['icon']} {name}"], x=[h], orientation='h',
            marker_color=health_color(h),
            text=f'{h:.0f}%', textposition='inside',
            textfont=dict(color='white', size=13, family='JetBrains Mono'),
            showlegend=False,
        ))
    fig_bar.update_layout(
        xaxis=dict(range=[0, 108], title="Health (%)"), yaxis=dict(autorange="reversed"),
        height=280, margin=dict(t=10, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.25,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── Row 3: Fault Chains + Impact ──
col_fault, col_impact = st.columns([1, 1])

with col_fault:
    st.markdown("#### 🔗 Active Fault Propagation Paths")

    if not active_chains:
        st.success("✅ No active fault propagation paths. All components within healthy limits.")
    else:
        for chain in active_chains[:5]:
            r = chain["risk"]
            css = "risk-critical" if r >= 75 else "risk-high" if r >= 50 else "risk-medium"
            st.markdown(f'''
            <div class="risk-card {css}">
                <div style="font-weight:600; margin-bottom:6px;">{risk_label(r)} Risk ({r:.0f}%) — {chain["trigger_component"]}</div>
                <div class="fault-path">{chain["path_display"]}</div>
                <div class="physics-note">{chain["physics"]}</div>
            </div>
            ''', unsafe_allow_html=True)

with col_impact:
    st.markdown("#### 💰 Impact Estimation")

    if sys_risk < 25:
        st.markdown('''
        <div style="background:#f0fff4; border-radius:12px; padding:24px; border:2px solid #c6f6d5;">
            <h4 style="color:#38a169; margin:0 0 8px 0;">✅ Low Risk — Normal Operations</h4>
            <p style="color:#4a5568; margin:0;">All components within acceptable limits. Continue routine monitoring schedule.</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        failure_prob = min(sys_risk / 100, 0.95)
        repair_cost = sum(
            (100 - h) / 100 * COMPONENTS[n]["replacement_cost"]
            for n, h in healths.items() if h < 70
        )
        downtime = max(4, int(sys_risk / 100 * 72))
        reg_penalty = int(sys_risk / 100 * 100000) if sys_risk > 60 else int(sys_risk / 100 * 10000)
        emergency_cost = pop_at_risk * 5
        total_cost = repair_cost + reg_penalty + emergency_cost

        st.markdown(f'''
        <div style="background:linear-gradient(135deg, #2d3748 0%, #1a202c 100%); border-radius:12px; padding:24px; color:white;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px;">Failure Probability</div>
                    <div style="font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono'; color:{risk_color(sys_risk)};">{failure_prob*100:.0f}%</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px;">Expected Downtime</div>
                    <div style="font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono';">{downtime}h</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px;">Repair Cost</div>
                    <div style="font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono';">${repair_cost:,.0f}</div>
                </div>
                <div>
                    <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px;">Regulatory Penalty</div>
                    <div style="font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono'; color:#fc8181;">${reg_penalty:,.0f}</div>
                </div>
            </div>
            <div style="margin-top:20px; border-top:1px solid #4a5568; padding-top:16px;">
                <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px;">Total Risk-Weighted Cost Exposure</div>
                <div style="font-size:2.8rem; font-weight:700; font-family:'JetBrains Mono'; color:#fbd38d;">${total_cost:,.0f}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("---")

# ── Row 4: KG Visualization ──
st.markdown("#### 🕸️ Knowledge Graph — Fault Propagation Network")
st.markdown("*Component health colors propagate through the causal chain from degradation to community impact.*")

kg_layers = {
    0: [("IGBT", "IGBT Module"), ("Capacitor", "DC-Link Capacitor"),
        ("Bearings", "Pump Bearings"), ("Seal", "Mechanical Seal")],
    1: [("Bond Wire\nLift-off", None), ("ESR\nDrift", None),
        ("Bearing\nWear", None), ("Seal\nDeterioration", None)],
    2: [("Vce_on\nRise", None), ("DC Bus\nRipple", None),
        ("Vibration ↑", None), ("Leakage", None)],
    3: [("Conduction\nLoss ↑", None), ("Harmonic\nDistortion", None),
        ("Thermal\nResistance ↑", None)],
    4: [("VFD Trip", None), ("Pump\nSeizure", None), ("Motor\nFailure", None)],
    5: [("Station\nOffline", None), ("Boil Water\nAdvisory", None), ("Emergency\nDistribution", None)],
}

layer_colors = {0: "#3182ce", 1: "#805ad5", 2: "#d69e2e", 3: "#dd6b20", 4: "#e53e3e", 5: "#1a1a2e"}

pos = {}
for layer, nodes in kg_layers.items():
    n = len(nodes)
    for i, (label, comp_key) in enumerate(nodes):
        x = layer * 2.0
        y = (i - (n - 1) / 2) * 1.6
        pos[label] = (x, y, layer, comp_key)

kg_edges = [
    ("IGBT", "Bond Wire\nLift-off"), ("Capacitor", "ESR\nDrift"),
    ("Bearings", "Bearing\nWear"), ("Seal", "Seal\nDeterioration"),
    ("Bond Wire\nLift-off", "Vce_on\nRise"), ("ESR\nDrift", "DC Bus\nRipple"),
    ("Bearing\nWear", "Vibration ↑"), ("Seal\nDeterioration", "Leakage"),
    ("Vce_on\nRise", "Conduction\nLoss ↑"), ("Vce_on\nRise", "Thermal\nResistance ↑"),
    ("DC Bus\nRipple", "Harmonic\nDistortion"),
    ("Conduction\nLoss ↑", "VFD Trip"), ("Thermal\nResistance ↑", "VFD Trip"),
    ("Harmonic\nDistortion", "VFD Trip"), ("Harmonic\nDistortion", "Motor\nFailure"),
    ("Vibration ↑", "Pump\nSeizure"), ("Leakage", "Motor\nFailure"),
    ("VFD Trip", "Station\nOffline"), ("Pump\nSeizure", "Station\nOffline"),
    ("Pump\nSeizure", "Boil Water\nAdvisory"), ("Motor\nFailure", "Boil Water\nAdvisory"),
    ("Motor\nFailure", "Emergency\nDistribution"),
]

fig_kg = go.Figure()

# Edges
for src, dst in kg_edges:
    x0, y0 = pos[src][0], pos[src][1]
    x1, y1 = pos[dst][0], pos[dst][1]
    # Highlight active edges
    src_layer = pos[src][2]
    is_active = False
    if src_layer == 0:
        comp_key = pos[src][3]
        if comp_key and healths.get(comp_key, 100) < 50:
            is_active = True
    fig_kg.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None], mode='lines',
        line=dict(width=2.5 if is_active else 1, color="#e53e3e" if is_active else "#cbd5e0"),
        hoverinfo='none', showlegend=False,
    ))

# Nodes
for label, (x, y, layer, comp_key) in pos.items():
    if layer == 0 and comp_key:
        color = health_color(healths.get(comp_key, 100))
        size = 32
    else:
        color = layer_colors.get(layer, "#666")
        size = 22 if layer < 4 else 28
    fig_kg.add_trace(go.Scatter(
        x=[x], y=[y], mode='markers+text', showlegend=False,
        marker=dict(size=size, color=color, line=dict(width=2, color='white')),
        text=[label], textposition="bottom center",
        textfont=dict(size=9, family="DM Sans", color="#2d3748"),
        hovertext=f"<b>{label}</b><br>Layer: {['Component','Degradation','Indicator','Subsystem','System Fault','Service Impact'][layer]}",
        hoverinfo='text',
    ))

# Layer labels
for i, lbl in enumerate(["Components", "Degradation\nModes", "Indicators",
                          "Subsystem\nEffects", "System\nFaults", "Service\nImpacts"]):
    fig_kg.add_annotation(x=i*2.0, y=4.0, text=f"<b>{lbl}</b>", showarrow=False,
                           font=dict(size=11, color="#4a5568"))

fig_kg.update_layout(
    height=420, showlegend=False,
    margin=dict(t=20, b=20, l=20, r=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
)
st.plotly_chart(fig_kg, use_container_width=True)

st.markdown("---")

# ── Row 5: RUL Projection + Maintenance Priority ──
col_rul, col_maint = st.columns([1, 1])

with col_rul:
    st.markdown("#### 📈 IGBT Degradation Projection (Sigmoid Model)")
    L, k, t0, c = SIGMOID_PARAMS["L"], SIGMOID_PARAMS["k"], SIGMOID_PARAMS["t0"], SIGMOID_PARAMS["c"]
    t_arr = np.linspace(0, 0.8, 300)
    vce_traj = L / (1 + np.exp(-k * (t_arr - t0))) + c
    threshold_v = c + 0.95 * L
    current_t = None
    for i, v in enumerate(vce_traj):
        if v >= vce_on:
            current_t = t_arr[i]
            break
    if current_t is None:
        current_t = 0

    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(x=t_arr, y=vce_traj, mode='lines', name='Predicted Path',
                                  line=dict(color='#3182ce', width=2.5)))
    fig_rul.add_hline(y=threshold_v, line_dash="dash", line_color="#e53e3e",
                       annotation_text=f"Failure ({threshold_v:.1f}V)")
    fig_rul.add_trace(go.Scatter(x=[current_t], y=[vce_on], mode='markers',
                                  name=f'Now (Vce={vce_on}V)',
                                  marker=dict(size=16, color=health_color(healths["IGBT Module"]),
                                              line=dict(width=2.5, color='white'), symbol='diamond')))
    # RUL annotation
    t_fail = t0 - (1.0 / k) * np.log((1.0 / 0.95) - 1.0)
    rul_remaining = max(0, t_fail - current_t)
    fig_rul.add_annotation(x=current_t, y=vce_on + 0.4,
                            text=f"RUL ≈ {rul_remaining:.3f}h", showarrow=True,
                            arrowhead=2, font=dict(size=12, color="#e53e3e", family="JetBrains Mono"))
    fig_rul.update_layout(
        xaxis_title="Aging Time (hours)", yaxis_title="Vce_on (V)",
        height=320, margin=dict(t=10, b=50, l=50, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        xaxis=dict(gridcolor="#edf2f7"), yaxis=dict(gridcolor="#edf2f7"),
    )
    st.plotly_chart(fig_rul, use_container_width=True)

with col_maint:
    st.markdown("#### 🛠️ Maintenance Priority")
    rows = []
    for name, h in sorted(healths.items(), key=lambda x: x[1]):
        urgency = 100 - h
        comp = COMPONENTS[name]
        action = "🔴 Replace Now" if h < 20 else "🟡 Schedule" if h < 50 else "🔵 Plan" if h < 70 else "🟢 Monitor"
        rows.append({
            "": comp["icon"],
            "Component": name,
            "Health": f"{h:.0f}%",
            "Urgency": f"{urgency:.0f}",
            "Cost": f"${comp['replacement_cost']:,}",
            "Lead": f"{comp['lead_time_days']}d",
            "Action": action,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=320)

st.markdown("---")

# ── Row 6: Preventive vs Reactive ──
st.markdown("#### 💡 Maintenance Decision: Prevent Now vs. Fix Later")

prev_cost = sum(COMPONENTS[n]["replacement_cost"] for n, h in healths.items() if h < 50)
react_cost = prev_cost * 3.5 + (int(sys_risk / 100 * 100000) if sys_risk > 50 else 0) + pop_at_risk * 5
downtime_prev = 12
downtime_react = max(8, int(sys_risk / 100 * 72))

c_left, c_mid, c_right = st.columns([5, 1, 5])

with c_left:
    st.markdown(f'''
    <div class="cost-box" style="background:#f0fff4; border:2px solid #c6f6d5;">
        <div class="cost-label" style="color:#38a169;">Preventive Maintenance (Now)</div>
        <div class="cost-value" style="color:#2f855a;">${prev_cost:,.0f}</div>
        <div class="cost-detail">
            Planned downtime: {downtime_prev} hours<br/>
            Regulatory risk: None<br/>
            Population impact: None<br/>
            Scheduled at convenience
        </div>
    </div>
    ''', unsafe_allow_html=True)

with c_mid:
    st.markdown("<div style='text-align:center; padding-top:60px; font-size:2rem; color:#a0aec0;'>vs</div>", unsafe_allow_html=True)

with c_right:
    st.markdown(f'''
    <div class="cost-box" style="background:#fff5f5; border:2px solid #fed7d7;">
        <div class="cost-label" style="color:#e53e3e;">Reactive Repair (After Failure)</div>
        <div class="cost-value" style="color:#c53030;">${react_cost:,.0f}</div>
        <div class="cost-detail">
            Emergency downtime: {downtime_react}+ hours<br/>
            Regulatory penalties: ${int(sys_risk/100*100000) if sys_risk>50 else 0:,.0f}<br/>
            Population affected: {pop_at_risk:,}<br/>
            Unplanned emergency response
        </div>
    </div>
    ''', unsafe_allow_html=True)

if prev_cost > 0 and react_cost > prev_cost:
    savings = react_cost - prev_cost
    roi = (savings / max(prev_cost, 1)) * 100
    st.markdown(f'''
    <div style="background:#ebf8ff; border-radius:12px; padding:16px; margin-top:16px; text-align:center; border:1px solid #bee3f8;">
        <span style="font-size:1.15rem; color:#2b6cb0; font-weight:600;">
            ✅ Preventive maintenance saves ${savings:,.0f} ({roi:.0f}% ROI) and protects {pop_at_risk:,} residents
        </span>
    </div>
    ''', unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")

with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    **Water Pump Infrastructure Risk Analytics** is a knowledge graph-driven prognostics framework
    developed at Mississippi State University. It bridges component-level Remaining Useful Life (RUL)
    estimation with system-level risk assessment for municipal water infrastructure.

    **Technical Foundation:**
    - IGBT RUL: Sigmoid degradation model with particle filter (NASA PCoE dataset, 4 devices)
    - Capacitor RUL: ESR drift and capacitance fade modeling
    - Knowledge Graph: 6-layer fault ontology from component physics to community impact
    - Risk Analytics: Weighted health aggregation with weakest-link penalty

    **Research Context:**
    Motivated by the 2022 Jackson, Mississippi water crisis, where aging pump infrastructure
    failed without adequate early warning, affecting 180,000 residents for 45+ days.

    **Framework:** PhD Dissertation — Dept. of ECE, Mississippi State University, 2026
    """)

st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.78rem; margin-top:8px;">
    Water Pump Infrastructure Risk Analytics &nbsp;|&nbsp; Knowledge Graph-Driven Prognostics Framework<br/>
    Department of Electrical and Computer Engineering &nbsp;|&nbsp; Mississippi State University &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)
