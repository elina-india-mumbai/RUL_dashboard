"""
Neo4j Knowledge Graph — IGBT/Capacitor Fault Propagation + Risk Analytics
==========================================================================
Run this ONCE to populate your Neo4j database.

Prerequisites:
    pip install neo4j
    Neo4j Desktop running with a local database (bolt://localhost:7687)

Usage:
    python populate_kg.py
"""

from neo4j import GraphDatabase
import sys

# ── CONFIGURE ───────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password_here"  # ← CHANGE THIS
# ────────────────────────────────────────────────────────────────


def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def create_constraints(tx):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DegradationMode) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Indicator) REQUIRE i.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SubsystemEffect) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:SystemFault) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (si:ServiceImpact) REQUIRE si.name IS UNIQUE",
    ]
    for c in constraints:
        try:
            tx.run(c)
        except Exception:
            pass


def create_components(tx):
    """Layer 0: Physical components in a VFD-driven pump system."""
    components = [
        # Power Electronics
        {"name": "IGBT Module", "category": "Power Electronics", "criticality": "Critical",
         "mtbf_hours": 50000, "replacement_cost": 2500, "lead_time_days": 14,
         "description": "Insulated Gate Bipolar Transistor — main switching device in VFD inverter stage"},
        {"name": "DC-Link Capacitor", "category": "Power Electronics", "criticality": "Critical",
         "mtbf_hours": 40000, "replacement_cost": 800, "lead_time_days": 7,
         "description": "Electrolytic capacitor in DC bus — energy storage and voltage smoothing"},
        {"name": "Gate Driver", "category": "Power Electronics", "criticality": "High",
         "mtbf_hours": 80000, "replacement_cost": 350, "lead_time_days": 5,
         "description": "Gate driver circuit providing switching signals to IGBT"},
        {"name": "Diode Bridge Rectifier", "category": "Power Electronics", "criticality": "High",
         "mtbf_hours": 100000, "replacement_cost": 200, "lead_time_days": 3,
         "description": "AC-DC rectifier stage of VFD"},
        # Mechanical
        {"name": "Pump Bearings", "category": "Mechanical", "criticality": "High",
         "mtbf_hours": 30000, "replacement_cost": 1200, "lead_time_days": 10,
         "description": "Radial and thrust bearings supporting pump shaft"},
        {"name": "Mechanical Seal", "category": "Mechanical", "criticality": "High",
         "mtbf_hours": 25000, "replacement_cost": 900, "lead_time_days": 7,
         "description": "Shaft seal preventing water ingress into motor cavity"},
        {"name": "Impeller", "category": "Mechanical", "criticality": "Medium",
         "mtbf_hours": 60000, "replacement_cost": 600, "lead_time_days": 14,
         "description": "Rotating element that moves water through the pump"},
        # Electrical
        {"name": "Stator Winding", "category": "Electrical", "criticality": "Critical",
         "mtbf_hours": 70000, "replacement_cost": 5000, "lead_time_days": 21,
         "description": "Motor stator winding — converts electrical to mechanical energy"},
    ]
    for c in components:
        tx.run("""
            CREATE (c:Component {
                name: $name, category: $category, criticality: $criticality,
                mtbf_hours: $mtbf_hours, replacement_cost: $replacement_cost,
                lead_time_days: $lead_time_days, description: $description,
                health_index: 100.0
            })
        """, **c)


def create_degradation_modes(tx):
    """Layer 1: How components degrade."""
    modes = [
        # IGBT degradation
        {"name": "Bond Wire Lift-off", "component": "IGBT Module",
         "physics": "CTE mismatch causes fatigue at wire-chip interface under thermal cycling",
         "detectable_early": True, "acceleration_factor": "temperature_swing"},
        {"name": "Solder Layer Fatigue", "component": "IGBT Module",
         "physics": "Creep and crack propagation in die-attach solder under thermal stress",
         "detectable_early": True, "acceleration_factor": "mean_temperature"},
        {"name": "Gate Oxide Degradation", "component": "IGBT Module",
         "physics": "Hot carrier injection traps charge in gate oxide, shifting threshold voltage",
         "detectable_early": False, "acceleration_factor": "gate_voltage_stress"},
        # Capacitor degradation
        {"name": "ESR Drift", "component": "DC-Link Capacitor",
         "physics": "Electrolyte evaporation through seal increases equivalent series resistance",
         "detectable_early": True, "acceleration_factor": "temperature"},
        {"name": "Capacitance Fade", "component": "DC-Link Capacitor",
         "physics": "Dielectric aging and electrolyte loss reduce charge storage capacity",
         "detectable_early": True, "acceleration_factor": "ripple_current"},
        # Mechanical degradation
        {"name": "Bearing Wear", "component": "Pump Bearings",
         "physics": "Surface fatigue, spalling, and lubricant degradation under cyclic loading",
         "detectable_early": True, "acceleration_factor": "vibration_load"},
        {"name": "Seal Deterioration", "component": "Mechanical Seal",
         "physics": "Elastomer aging, face wear, and thermal degradation of O-rings",
         "detectable_early": False, "acceleration_factor": "temperature_chemical"},
        # Electrical degradation
        {"name": "Insulation Breakdown", "component": "Stator Winding",
         "physics": "Partial discharge erodes enamel insulation under voltage stress",
         "detectable_early": True, "acceleration_factor": "voltage_harmonics"},
    ]
    for m in modes:
        tx.run("""
            MATCH (c:Component {name: $component})
            CREATE (d:DegradationMode {
                name: $name, physics: $physics,
                detectable_early: $detectable_early,
                acceleration_factor: $acceleration_factor
            })
            CREATE (c)-[:EXPERIENCES]->(d)
        """, **m)


def create_indicators(tx):
    """Layer 2: Measurable degradation indicators with RUL model parameters."""
    indicators = [
        # IGBT indicators
        {"name": "Vce_on Rise", "unit": "V", "healthy_range": "1.2-1.5",
         "failure_threshold_description": "95% of sigmoid saturation",
         "degradation_mode": "Bond Wire Lift-off",
         "model_type": "sigmoid", "model_L": 4.456, "model_k": 11.084,
         "model_t0": 0.0778, "model_c": 0.0,
         "rul_method": "Particle Filter (1000 particles, adaptive threshold)"},
        {"name": "Switching Time Increase", "unit": "us", "healthy_range": "0.5-2.0",
         "failure_threshold_description": "200% of initial value",
         "degradation_mode": "Solder Layer Fatigue",
         "model_type": "linear_trend", "model_L": 0, "model_k": 0,
         "model_t0": 0, "model_c": 0, "rul_method": "Trend extrapolation"},
        {"name": "Vge_th Shift", "unit": "V", "healthy_range": "4.5-5.5",
         "failure_threshold_description": "Outside ±15% of nominal",
         "degradation_mode": "Gate Oxide Degradation",
         "model_type": "linear_drift", "model_L": 0, "model_k": 0,
         "model_t0": 0, "model_c": 0, "rul_method": "Monitoring only"},
        {"name": "Ic_peak Decline", "unit": "A", "healthy_range": "7.5-9.0",
         "failure_threshold_description": "Below 70% of rated current",
         "degradation_mode": "Bond Wire Lift-off",
         "model_type": "correlated_with_Vce", "model_L": 0, "model_k": 0,
         "model_t0": 0, "model_c": 0, "rul_method": "Secondary indicator"},
        # Capacitor indicators
        {"name": "ESR Increase", "unit": "mOhm", "healthy_range": "10-50",
         "failure_threshold_description": "200% of initial ESR",
         "degradation_mode": "ESR Drift",
         "model_type": "exponential", "model_L": 0, "model_k": 0.001,
         "model_t0": 0, "model_c": 15.0, "rul_method": "Exponential fit + PF"},
        {"name": "Capacitance Loss", "unit": "uF", "healthy_range": "900-1100",
         "failure_threshold_description": "Below 80% of rated capacitance",
         "degradation_mode": "Capacitance Fade",
         "model_type": "linear_decay", "model_L": 0, "model_k": 0,
         "model_t0": 0, "model_c": 1000.0, "rul_method": "Linear projection"},
        # Mechanical indicators
        {"name": "Vibration Amplitude", "unit": "mm/s", "healthy_range": "0-4.5",
         "failure_threshold_description": "ISO 10816 Zone D (>11.2 mm/s)",
         "degradation_mode": "Bearing Wear",
         "model_type": "exponential", "model_L": 0, "model_k": 0,
         "model_t0": 0, "model_c": 2.0, "rul_method": "Vibration trending"},
        # Electrical indicators
        {"name": "Partial Discharge Level", "unit": "pC", "healthy_range": "0-100",
         "failure_threshold_description": "Above 1000 pC sustained",
         "degradation_mode": "Insulation Breakdown",
         "model_type": "threshold", "model_L": 0, "model_k": 0,
         "model_t0": 0, "model_c": 0, "rul_method": "PD monitoring"},
    ]
    for ind in indicators:
        tx.run("""
            MATCH (d:DegradationMode {name: $degradation_mode})
            CREATE (i:Indicator {
                name: $name, unit: $unit, healthy_range: $healthy_range,
                failure_threshold_description: $failure_threshold_description,
                model_type: $model_type, model_L: $model_L, model_k: $model_k,
                model_t0: $model_t0, model_c: $model_c, rul_method: $rul_method
            })
            CREATE (d)-[:MANIFESTS_AS]->(i)
        """, **ind)


def create_subsystem_effects(tx):
    """Layer 3: What happens at subsystem level when components degrade."""
    effects = [
        {"name": "Increased Conduction Loss", "subsystem": "VFD Inverter",
         "description": "Rising Vce_on increases I²R losses in IGBT, generating excess heat",
         "indicator": "Vce_on Rise", "severity": "Progressive",
         "observable_by_scada": True},
        {"name": "DC Bus Voltage Ripple", "subsystem": "VFD DC Bus",
         "description": "Capacitor ESR rise and capacitance loss increase ripple on DC bus",
         "indicator": "ESR Increase", "severity": "Progressive",
         "observable_by_scada": True},
        {"name": "Output Harmonic Distortion", "subsystem": "VFD Output",
         "description": "Degraded switching performance produces distorted motor current waveforms",
         "indicator": "Switching Time Increase", "severity": "Progressive",
         "observable_by_scada": True},
        {"name": "Thermal Resistance Rise", "subsystem": "VFD Thermal Management",
         "description": "Solder degradation reduces heat transfer from die to heatsink",
         "indicator": "Vce_on Rise", "severity": "Progressive",
         "observable_by_scada": False},
        {"name": "Motor Efficiency Loss", "subsystem": "Pump Motor",
         "description": "Harmonic currents cause additional copper and iron losses in motor",
         "indicator": "Switching Time Increase", "severity": "Progressive",
         "observable_by_scada": True},
        {"name": "Bearing Vibration Increase", "subsystem": "Pump Mechanical",
         "description": "Wear particles and surface damage increase vibration signature",
         "indicator": "Vibration Amplitude", "severity": "Progressive",
         "observable_by_scada": True},
        {"name": "Seal Leakage", "subsystem": "Pump Sealing",
         "description": "Worn seal faces allow water ingress into motor housing",
         "indicator": "Vibration Amplitude", "severity": "Step-change",
         "observable_by_scada": False},
    ]
    for e in effects:
        tx.run("""
            MATCH (i:Indicator {name: $indicator})
            CREATE (s:SubsystemEffect {
                name: $name, subsystem: $subsystem, description: $description,
                severity: $severity, observable_by_scada: $observable_by_scada
            })
            CREATE (i)-[:CAUSES]->(s)
        """, **e)


def create_system_faults(tx):
    """Layer 4: System-level fault events."""
    faults = [
        {"name": "VFD Overcurrent Trip", "fault_code": "F001",
         "description": "VFD protection shuts down on excess current draw",
         "mttr_hours": 4, "requires_specialist": True,
         "effects": ["Increased Conduction Loss", "DC Bus Voltage Ripple"]},
        {"name": "VFD Overtemperature Shutdown", "fault_code": "F002",
         "description": "VFD thermal protection triggers on heatsink temperature exceedance",
         "mttr_hours": 2, "requires_specialist": False,
         "effects": ["Thermal Resistance Rise", "Increased Conduction Loss"]},
        {"name": "VFD DC Bus Fault", "fault_code": "F003",
         "description": "DC bus overvoltage or undervoltage protection activates",
         "mttr_hours": 6, "requires_specialist": True,
         "effects": ["DC Bus Voltage Ripple"]},
        {"name": "Motor Insulation Failure", "fault_code": "F004",
         "description": "Phase-to-ground or phase-to-phase short in motor winding",
         "mttr_hours": 48, "requires_specialist": True,
         "effects": ["Output Harmonic Distortion", "Motor Efficiency Loss"]},
        {"name": "Pump Seizure", "fault_code": "F005",
         "description": "Bearing failure causes shaft lockup, motor overcurrent trip",
         "mttr_hours": 24, "requires_specialist": True,
         "effects": ["Bearing Vibration Increase"]},
        {"name": "Water Ingress to Motor", "fault_code": "F006",
         "description": "Seal failure allows water into motor housing, causing insulation damage",
         "mttr_hours": 72, "requires_specialist": True,
         "effects": ["Seal Leakage"]},
    ]
    for f in faults:
        tx.run("""
            CREATE (sf:SystemFault {
                name: $name, fault_code: $fault_code, description: $description,
                mttr_hours: $mttr_hours, requires_specialist: $requires_specialist
            })
        """, name=f["name"], fault_code=f["fault_code"],
             description=f["description"], mttr_hours=f["mttr_hours"],
             requires_specialist=f["requires_specialist"])
        for eff in f["effects"]:
            tx.run("""
                MATCH (se:SubsystemEffect {name: $effect})
                MATCH (sf:SystemFault {name: $fault})
                CREATE (se)-[:LEADS_TO {probability: 0.7}]->(sf)
            """, effect=eff, fault=f["name"])


def create_service_impacts(tx):
    """Layer 5: Service-level consequences for the community."""
    impacts = [
        {"name": "Pump Station Offline", "category": "Infrastructure",
         "description": "Complete loss of pumping capacity at the station",
         "faults": ["VFD Overcurrent Trip", "VFD Overtemperature Shutdown",
                     "VFD DC Bus Fault", "Pump Seizure", "Water Ingress to Motor"],
         "population_affected": 12000, "duration_hours": 8,
         "direct_cost": 15000, "regulatory_penalty": 5000},
        {"name": "Reduced Water Pressure", "category": "Service Quality",
         "description": "Partial capacity loss reducing system pressure below minimum",
         "faults": ["VFD Overcurrent Trip", "Motor Insulation Failure"],
         "population_affected": 8000, "duration_hours": 4,
         "direct_cost": 5000, "regulatory_penalty": 2000},
        {"name": "Boil Water Advisory", "category": "Public Health",
         "description": "Loss of pressure triggers mandatory boil water notice",
         "faults": ["Pump Seizure", "Water Ingress to Motor"],
         "population_affected": 25000, "duration_hours": 72,
         "direct_cost": 50000, "regulatory_penalty": 100000},
        {"name": "Treatment Bypass", "category": "Environmental",
         "description": "Untreated or partially treated water released due to pump failure",
         "faults": ["Pump Seizure", "VFD DC Bus Fault"],
         "population_affected": 50000, "duration_hours": 24,
         "direct_cost": 200000, "regulatory_penalty": 500000},
        {"name": "Emergency Water Distribution", "category": "Emergency Response",
         "description": "Bottled water distribution required for affected population",
         "faults": ["Pump Seizure", "Water Ingress to Motor", "Motor Insulation Failure"],
         "population_affected": 15000, "duration_hours": 48,
         "direct_cost": 75000, "regulatory_penalty": 0},
    ]
    for imp in impacts:
        tx.run("""
            CREATE (si:ServiceImpact {
                name: $name, category: $category, description: $description,
                population_affected: $population_affected,
                duration_hours: $duration_hours,
                direct_cost: $direct_cost, regulatory_penalty: $regulatory_penalty,
                total_cost: $direct_cost + $regulatory_penalty
            })
        """, name=imp["name"], category=imp["category"],
             description=imp["description"],
             population_affected=imp["population_affected"],
             duration_hours=imp["duration_hours"],
             direct_cost=imp["direct_cost"],
             regulatory_penalty=imp["regulatory_penalty"])
        for fault in imp["faults"]:
            tx.run("""
                MATCH (sf:SystemFault {name: $fault})
                MATCH (si:ServiceImpact {name: $impact})
                CREATE (sf)-[:RESULTS_IN]->(si)
            """, fault=fault, impact=imp["name"])


def create_pump_stations(tx):
    """Create sample pump station nodes (Jackson-inspired)."""
    stations = [
        {"name": "Pump Station North", "location": "North Zone",
         "capacity_mgd": 4.5, "age_years": 18, "redundancy": "N+1",
         "population_served": 15000},
        {"name": "Pump Station South", "location": "South Zone",
         "capacity_mgd": 6.0, "age_years": 25, "redundancy": "None",
         "population_served": 22000},
        {"name": "Pump Station Central", "location": "Central Zone",
         "capacity_mgd": 8.0, "age_years": 12, "redundancy": "N+1",
         "population_served": 35000},
    ]
    for s in stations:
        tx.run("""
            CREATE (ps:PumpStation {
                name: $name, location: $location, capacity_mgd: $capacity_mgd,
                age_years: $age_years, redundancy: $redundancy,
                population_served: $population_served,
                risk_score: 0.0
            })
        """, **s)
        # Link each station to components
        tx.run("""
            MATCH (ps:PumpStation {name: $station})
            MATCH (c:Component)
            CREATE (ps)-[:CONTAINS]->(c)
        """, station=s["name"])


def create_jackson_scenario(tx):
    """Add Jackson crisis reference data for validation."""
    tx.run("""
        CREATE (j:CaseStudy {
            name: "Jackson Water Crisis 2022",
            location: "Jackson, Mississippi",
            population_affected: 180000,
            duration_days: 45,
            root_causes: "Aging infrastructure, deferred maintenance, freezing weather",
            total_cost_estimate: 200000000,
            key_failure: "O.B. Curtis Water Treatment Plant pump failures",
            lesson: "Component degradation without monitoring led to cascading system failure"
        })
    """)


def main():
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connected successfully.\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        print(f"\nMake sure Neo4j is running and update credentials in this script.")
        print(f"Current settings: URI={NEO4J_URI}, User={NEO4J_USER}")
        sys.exit(1)

    with driver.session() as session:
        print("Clearing existing data...")
        session.execute_write(clear_database)

        print("Creating constraints...")
        session.execute_write(create_constraints)

        print("Creating components...")
        session.execute_write(create_components)

        print("Creating degradation modes...")
        session.execute_write(create_degradation_modes)

        print("Creating indicators...")
        session.execute_write(create_indicators)

        print("Creating subsystem effects...")
        session.execute_write(create_subsystem_effects)

        print("Creating system faults...")
        session.execute_write(create_system_faults)

        print("Creating service impacts...")
        session.execute_write(create_service_impacts)

        print("Creating pump stations...")
        session.execute_write(create_pump_stations)

        print("Creating Jackson case study...")
        session.execute_write(create_jackson_scenario)

        # Verify
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label")
        print("\n" + "=" * 50)
        print("KNOWLEDGE GRAPH POPULATED SUCCESSFULLY")
        print("=" * 50)
        for record in result:
            print(f"  {record['label']:20s}: {record['count']} nodes")

        result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY type")
        print("\nRelationships:")
        for record in result:
            print(f"  {record['type']:20s}: {record['count']} edges")

    driver.close()
    print("\nDone. Open Neo4j Browser and run:  MATCH (n)-[r]->(m) RETURN n,r,m")


if __name__ == "__main__":
    main()
