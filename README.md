# 🔧 Water Pump Infrastructure Risk Analytics

**Knowledge Graph-Driven Prognostics for Municipal Water Systems**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Overview

This dashboard bridges **component-level Remaining Useful Life (RUL) estimation** with **system-level risk assessment** for municipal water pump infrastructure. It translates engineering prognostics into actionable risk intelligence for utility operators, city managers, and stakeholders.

Motivated by the **2022 Jackson, Mississippi water crisis**, where aging pump infrastructure failed without adequate early warning, affecting 180,000 residents for 45+ days.

## Features

- **Component Health Monitoring** — Real-time health index (0–100) for IGBTs, capacitors, bearings, seals
- **Fault Propagation Visualization** — 6-layer knowledge graph showing causal chains from component degradation to community impact
- **Risk Scoring** — System-level risk aggregation with weighted component contributions
- **Impact Estimation** — Cost exposure, regulatory penalties, population affected, expected downtime
- **Maintenance Decision Support** — Preventive vs. reactive cost comparison with ROI calculation
- **RUL Projection** — Sigmoid degradation model with real-time remaining life estimation
- **Scenario Simulation** — Preset scenarios including Jackson Crisis simulation

## Technical Foundation

| Layer | Method | Data Source |
|-------|--------|-------------|
| IGBT RUL | Sigmoid model + Particle Filter | NASA PCoE Accelerated Aging |
| Capacitor RUL | ESR drift + Capacitance fade | NASA PCoE Dataset |
| Knowledge Graph | 6-layer fault ontology | Physics of failure + domain expertise |
| Risk Analytics | Weighted health + weakest-link | Monte Carlo validated |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/water-pump-risk-analytics.git
cd water-pump-risk-analytics

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy

No database or external services required — the knowledge graph is embedded.

## Repository Structure

```
├── app.py                 # Main Streamlit dashboard
├── requirements.txt       # Python dependencies
├── populate_kg.py         # Neo4j population script (optional, for local Neo4j)
└── README.md
```

## Research Context

**PhD Dissertation** — Department of Electrical and Computer Engineering, Mississippi State University, 2026

**Framework:** Bottom-up component RUL → Knowledge Graph fault propagation → System-level risk analytics → Community impact assessment

---

*Built with Streamlit, Plotly, and NetworkX*
