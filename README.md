# Satellite Sensor Simulation

A Python-based simulation framework for modeling satellite-based methane detection sensors.

## Overview

This project provides tools to simulate and analyze:

- Radiometric performance
- Quantum efficiency interpolation
- Photon count estimation
- Atmospheric transmission modeling
- Attitude control system calculations

## Project Structure

- `00_data/` - Contains quantum efficiency and spectral response data
- `02_scripts/` - Core simulation modules
  - `attitude_control_system.py` - AOCS calculations
  - `interpolate_qe.py` - Quantum efficiency interpolation
  - `mathematical_modelling.py` - Core radiometric equations
  - `photon_count.py` - Photon statistics modeling
  - `radianceAtSensor.py` - Top-of-atmosphere radiance calculations
  - `total_at_aparture_radiance_model.py` - Aperture radiance model
- `03_tests/` - Test scripts and validation tools

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install numpy scipy pandas matplotlib Py6S
```

## Example Usage

```python
from mathematical_modelling import TheorySec2

# Initialize simulation
theory = TheorySec2(
    altitude_m=500e3,          # Altitude in meters
    t_int=400,                 # Integration time in seconds
    temperature=20,            # Temperature in Celsius
    lambda_=1662.2,           # Wavelength for methane detection in nm
)

# Run calculations
results = theory.run_simulation()
```
## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
