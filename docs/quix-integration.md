# QuiX Quantum Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Profile Configuration](#profile-configuration)
5. [Backend: QuiXCloudBackend](#backend-quixcloudbackend)
6. [Data Center Simulator](#data-center-simulator)
7. [PUE Energy Model](#pue-energy-model)
8. [Demo Scenarios](#demo-scenarios)
9. [API Usage](#api-usage)
10. [Production Deployment](#production-deployment)

---

## 1. Overview

QuiX Quantum is a Dutch photonic quantum computing company building silicon-nitride-based universal quantum computers designed for data-center integration. Unlike edge-focused deployments (Rotonium), QuiX targets:

- **HPC clusters** for scientific computing (pharma, materials science)
- **Data center racks** for enterprise workloads (finance, logistics)
- **Cloud nodes** for on-demand quantum access (via cloud.quixquantum.com)

The QuantumEdge Pipeline supports QuiX through the **Target Profiles** architecture, allowing the same codebase to serve both Rotonium edge and QuiX data-center use cases.

### Key Differentiators

| Feature | Rotonium (Edge) | QuiX Quantum (Data Center) |
|---------|----------------|---------------------------|
| Deployment | Aerospace, mobile, ground station | HPC, rack, cloud |
| Energy Model | SWaP (Size, Weight, Power) | PUE (Power Usage Effectiveness) |
| Key Metric | Battery budget | Throughput / cost per job |
| Hardware | OAM photonic QPU (simulated) | Silicon nitride photonic (real API) |
| Temperature | Room temperature | Room temperature |
| Cryogenics | None required | None required |

---

## 2. Architecture

```
                    ┌─────────────────────────┐
                    │     profiles/quix.yaml   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    src/profile_loader.py  │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼─────────┐ ┌─────▼──────┐  ┌────────▼────────┐
    │ QuiXCloudBackend   │ │ Dashboard  │  │ FastAPI API      │
    │ src/backends/      │ │ Branding   │  │ /system/info     │
    │ quix_cloud.py      │ │ Scenarios  │  │ Profile metadata │
    └─────────┬─────────┘ │ Metrics    │  └─────────────────┘
              │            └────────────┘
              │
    ┌─────────▼─────────────────────┐
    │ DatacenterEnvironment          │
    │ src/router/datacenter_simulator│
    │ HPC / Rack / Cloud profiles   │
    └───────────────────────────────┘
```

---

## 3. Quick Start

### Switch to QuiX profile

```bash
# Via environment variable
QUANTUMEDGE_PROFILE=quix streamlit run dashboard/app.py

# Via CLI argument
streamlit run dashboard/app.py -- --profile quix

# Via Docker
QUANTUMEDGE_PROFILE=quix docker-compose up -d
```

### Connect to real QuiX hardware

```bash
export QUIX_API_KEY="your-api-key-here"
QUANTUMEDGE_PROFILE=quix python -m src.api.main
```

Without `QUIX_API_KEY`, the backend runs in **mock mode** producing simulated results with realistic response shapes.

---

## 4. Profile Configuration

The QuiX profile is defined in `profiles/quix.yaml`:

```yaml
name: "QuiX Quantum"
tagline: "Data Center Quantum Computing - Silicon Nitride Photonic"
hardware_backend: "quix_cloud"

deployment_profiles:
  primary: "hpc_cluster"
  available: ["hpc_cluster", "datacenter_rack", "cloud_node"]

routing:
  strategy_default: "cost_per_job"
  key_metric: "throughput"
  power_unit: "pue_adjusted"

energy_model:
  framing: "PUE"
  label: "Data Center Energy Efficiency"
  warn_threshold_pct: 90
```

---

## 5. Backend: QuiXCloudBackend

Located at `src/backends/quix_cloud.py`, this backend implements the `QuantumBackend` ABC with two modes:

### Mock Mode (default)
When no `QUIX_API_KEY` is set, the backend generates realistic simulated results locally. This is suitable for demos, testing, and development.

### Real Mode
When `QUIX_API_KEY` is set, the backend connects to `https://cloud.quixquantum.com/api` using `httpx` for async HTTP requests.

```python
from src.backends import create_backend

# Mock mode
backend = create_backend("quix_cloud")

# Real mode
backend = create_backend("quix_cloud", api_key="your-key")

# Submit a job
job_id = backend.submit_job(circuit, shots=1024)
result = backend.get_job_result(job_id)
```

### Hardware Specs

```python
>>> backend.get_hardware_specs()
{
    "provider": "QuiX Quantum",
    "technology": "Silicon Nitride Photonic",
    "max_qubits": 20,
    "clock_speed_mhz": 100,
    "optical_loss_db_per_cm": 0.1,
    "circuit_fidelity_pct": 99.0,
    "cryogenics_required": False,
    ...
}
```

---

## 6. Data Center Simulator

The `DatacenterEnvironment` in `src/router/datacenter_simulator.py` models three deployment tiers:

| Profile | Power | Memory | Timeout | GPU | Network | PUE |
|---------|-------|--------|---------|-----|---------|-----|
| HPC Cluster | 5000 W | 512 GB | 3600 s | 8 | 100 Gbps | 1.2 |
| Datacenter Rack | 2000 W | 128 GB | 600 s | 2 | 25 Gbps | 1.4 |
| Cloud Node | 500 W | 64 GB | 300 s | 0 | 10 Gbps | 1.3 |

```python
from src.router.datacenter_simulator import DatacenterEnvironment, DatacenterProfile

env = DatacenterEnvironment(DatacenterProfile.HPC_CLUSTER)
print(env.get_profile_info())
```

---

## 7. PUE Energy Model

Power Usage Effectiveness (PUE) accounts for data-center overhead:

```
PUE = Total Facility Power / IT Equipment Power
```

- **PUE 1.0** = perfect efficiency (impossible)
- **PUE 1.2** = excellent (modern HPC, Google-class)
- **PUE 1.4** = average data center
- **PUE 2.0** = inefficient legacy facility

The QuiX energy model applies PUE to all energy estimates:

```python
raw_energy_mj = 10.0
pue_adjusted = env.calculate_pue_adjusted_energy(raw_energy_mj)
# With PUE 1.3: pue_adjusted = 13.0 mJ
```

This contrasts with Rotonium's SWaP (Size, Weight, Power) model which focuses on battery budget for edge devices.

---

## 8. Demo Scenarios

Three QuiX-specific scenarios are included:

| Scenario | Problem | Size | Profile | Source |
|----------|---------|------|---------|--------|
| Drug Discovery Sampling | MaxCut | 50 | HPC Cluster | `examples/scenarios/quix/pharma_optimization.json` |
| Financial Risk Modeling | Portfolio | 40 | Datacenter Rack | `examples/scenarios/quix/portfolio_risk.json` |
| Hydrology Simulation | MaxCut | 45 | Cloud Node | `examples/scenarios/quix/hydrology.json` |

These scenarios are automatically shown in the dashboard when the QuiX profile is active.

---

## 9. API Usage

The API exposes profile information at `/api/v1/system/info`:

```bash
curl http://localhost:8000/api/v1/system/info | jq '.configuration.active_profile'
```

```json
{
  "name": "QuiX Quantum",
  "tagline": "Data Center Quantum Computing - Silicon Nitride Photonic",
  "hardware_backend": "quix_cloud",
  "deployment_profiles": ["hpc_cluster", "datacenter_rack", "cloud_node"],
  "energy_model": "PUE"
}
```

---

## 10. Production Deployment

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `QUANTUMEDGE_PROFILE` | No | Set to `quix` to activate QuiX profile (default: `default`) |
| `QUIX_API_KEY` | For real hardware | API key from cloud.quixquantum.com |

### Docker Compose

```bash
# Deploy with QuiX profile
QUANTUMEDGE_PROFILE=quix QUIX_API_KEY=your-key docker-compose up -d
```

### Security Notes

- Never commit `QUIX_API_KEY` to version control
- Use `.env` file or secrets manager for API keys
- The `QuiXCloudBackend` gracefully falls back to mock mode if no key is set
