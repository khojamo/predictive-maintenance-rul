from __future__ import annotations

# Canonical, user-friendly feature names for industrial assets.
# These are used for mapping uploaded datasets to a common schema.

ID_COL = "unit_id"
TIME_COL = "cycle"

LABEL_RUL = "rul"
LABEL_FAIL_WITHIN_H = "fail_within_h"
LABEL_FAILURE_CYCLE = "failure_cycle"

CANONICAL_FEATURES: list[str] = [
    "operating_mode",
    "ambient_temperature",
    "ambient_pressure",
    "humidity",
    "intake_air_temperature",
    "intake_air_pressure",
    "inlet_flow_rate",
    "fuel_flow_rate",
    "coolant_flow_rate",
    "oil_pressure",
    "oil_temperature",
    "oil_viscosity",
    "coolant_temperature",
    "coolant_pressure",
    "exhaust_gas_temperature",
    "exhaust_pressure",
    "bearing_temperature",
    "stator_temperature",
    "rotor_temperature",
    "casing_temperature",
    "vibration_rms",
    "vibration_peak",
    "vibration_kurtosis",
    "vibration_skewness",
    "axial_vibration",
    "radial_vibration",
    "shaft_speed_rpm",
    "shaft_torque",
    "power_output_kw",
    "motor_current",
    "motor_voltage",
    "power_factor",
    "energy_consumption",
    "load_percent",
    "pressure_ratio",
    "differential_pressure",
    "flow_rate",
    "valve_position",
    "actuator_position",
    "control_signal",
    "setpoint",
    "error_signal",
    "efficiency",
    "heat_rate",
    "discharge_pressure",
    "suction_pressure",
    "runtime_hours",
]

CANONICAL_LABELS: list[str] = [LABEL_RUL, LABEL_FAIL_WITHIN_H, LABEL_FAILURE_CYCLE]
