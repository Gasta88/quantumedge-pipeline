# QuantumEdge Pipeline

> **Status Legend**: ✅ = Fully Implemented & Tested | 🚧 = Interface Ready, Requires External Resources/SDK

## Overview

### What is QuantumEdge Pipeline?

**QuantumEdge Pipeline** is a quantum-classical hybrid optimization framework designed specifically for **edge computing environments**. It routes computational workloads between classical and quantum solvers based on real-time problem analysis, resource constraints, and performance requirements.

### Why It Matters (Dual Target Profiles)

Organizations need **practical tools** to evaluate, integrate, and deploy quantum solutions across very different environments. QuantumEdge Pipeline now supports multiple company targets via profile-driven configuration:

-  **Proving viability early**: Demonstrates quantum advantage in both edge and data-center scenarios before hardware is available
-  **Reducing integration time**: Provides APIs and workflows for quantum-classical hybrid systems with pluggable backends
-  **Profile-aware architecture**: Optimized for the active deployment profile (edge SWaP vs. data-center PUE)
-  **Hardware-agnostic design**: Seamlessly transitions from simulation to real quantum hardware through the new backend interface

For **Rotonium**, this pipeline showcases how their photonic quantum processors can be integrated into edge deployments, particularly for:
- NATO DIANA defense applications
- Aerospace optimization (flight routing, trajectory planning)
- Space-based quantum computing scenarios
- Room-temperature quantum operations at the edge

For **QuiX Quantum**, the pipeline highlights data-center and HPC integrations:
- Pharma and drug discovery workloads that benefit from photonic sampling
- Financial risk modeling within enterprise racks or cloud nodes
- Hydrology and climate simulations co-located with HPC resources
- Drop-in rack deployments leveraging silicon-nitride photonic processors

###  Key Features

- **Problem Analysis**: Automatic characterization and routing of optimization problems
- **Solver Selection**: Dynamic decision-making between classical, quantum, and hybrid approaches
- **Comparative Benchmarking**: Side-by-side performance evaluation of multiple solver strategies
- **Edge-Optimized**: Designed for SWaP (Size, Weight, and Power) constrained environments
- **Accessible API**: FastAPI-based RESTful interface with OpenAPI documentation
- **Real-Time Monitoring**: Comprehensive metrics, dashboards, and performance tracking
- **Docker-First Deployment**: Complete containerized infrastructure for rapid deployment

### Dual Target Profiles

The pipeline can now morph between company-specific configurations via **Target Profiles**:

| Profile | Use Case | Backend | Energy Model | Demo Scenarios |
|---------|----------|---------|--------------|----------------|
| `rotonium` (default) | Edge, aerospace, defense | `rotonium_mock` (photonic simulator) | SWaP (battery budget) | UAV routing, satellite optimization, edge AI |
| `quix` | HPC, data center, cloud | `quix_cloud` (real API + mock fallback) | PUE (data-center efficiency) | Pharma sampling, portfolio risk, hydrology |

Select a profile via CLI or environment variable:

```bash
# CLI flag (Streamlit)
streamlit run dashboard/app.py -- --profile quix

# Environment variable (API + dashboard + Docker)
export QUANTUMEDGE_PROFILE=quix
export QUIX_API_KEY="your-real-api-key"   # Optional; enables real hardware access
```

Profiles live in the new `profiles/` directory and are loaded by `src/profile_loader.py`. Each profile controls branding, deployment options, demo scenarios, docs links, and which backend implementation is instantiated.

---

##  Quick Start

For use of Makefile commands, see [**docs/quickstart.md**](docs/quickstart.md)

### Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Python** 3.9+ (for local development)
- **Poetry** (for dependency management)
- **Git** (for version control)

### Installation (Docker-based)

```bash
# 1. Clone the repository
git clone https://github.com/Gasta88/quantumedge-pipeline.git
cd quantumedge-pipeline

# 2. Copy environment configuration and configure it
cp .env.example .env

# 3. (Optional) Select company profile
#    Choices: rotonium (default), quix
export QUANTUMEDGE_PROFILE=quix

# Provide QuiX API key only if you have real hardware access
# export QUIX_API_KEY=your-key

# 4. Start all services with Docker Compose
make up

# 5. Verify services are running
make ps

# 6. In case of unhealthy services, restart them
make restart
```

### Access Dashboards

Once services are running:

| Service | URL | Description |
|---------|-----|-------------|
| **Interactive Dashboard** | http://localhost:8501 | Streamlit-based UI for problem submission and visualization |
| **API Documentation** | http://localhost:8000/docs | OpenAPI/Swagger interactive API docs |
| **API Health Check** | http://localhost:8000/health | System status and health monitoring |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANTUMEDGE_PROFILE` | `default` (Rotonium) | Selects the active company profile (`rotonium`, `quix`, or custom) |
| `QUIX_API_KEY` | _empty_ | Optional. Enables real QuiX cloud hardware access; absent → mock mode |

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────────────┐            ┌─────────────────────────┐    │
│  │  Streamlit       │            │   REST API Clients      │    │
│  │  Dashboard       │            │   (curl, Python, etc.)  │    │
│  └────────┬─────────┘            └────────┬────────────────┘    │
└───────────┼───────────────────────────────┼─────────────────────┘
            │                               │
            └──────────────┬────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│                     FastAPI Server                            │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              Optimization Orchestrator               │     │
│  └───┬──────────────────────────────────────────────┬───┘     │
│      │                                              │         │
│  ┌───▼─────────────┐                        ┌──────▼──────┐   │
│  │  Problem        │                        │   Metrics   │   │
│  │  Analyzer       │                        │  Collector  │   │
│  └───┬─────────────┘                        └─────────────┘   │
│      │                                                        │
│  ┌───▼─────────────────────────────────────────────────────┐  │
│  │          Quantum Router                                 │  │
│  │  (ML-based routing decision engine)                     │  │
│  └───┬─────────────────────┬─────────────────────┬─────────┘  │
└──────┼─────────────────────┼─────────────────────┼────────────┘
       │                     │                     │
   ┌───▼────────┐      ┌─────▼──────────┐   ┌──────▼───────┐
   │ Classical  │      │    Quantum     │   │   Hybrid     │
   │ Solvers    │      │    Solvers     │   │   Solvers ✅ │
   ├────────────┤      ├────────────────┤   └──────────────┘
   │• Greedy ✅ │      │• QAOA ✅       │          │
   │• SimAnneal✅│     │• VQE ✅        │  ┌───────▼────────┐
   │• OR-Tools✅│      │• PennyLane ✅  │  │   PostgreSQL   │
   │• SciPy ✅  │      │• IBM Quantum🚧 │  │   (Metrics &   │
   │• Gurobi 🚧 │      │• AWS Braket 🚧 │  │   Job History) │
   │• NetworkX🚧│      │• Rotonium QPU🚧│  └────────────────┘
   └────────────┘      └────────────────┘
```

### Component Descriptions

#### 1. **Problem Analyzer** (`src/analyzer/`)
- Extracts structural features from optimization problems
- Computes complexity metrics (graph density, connectivity, etc.)
- Generates feature vectors for routing decisions
- Supports: MaxCut, TSP, Portfolio Optimization, Graph Partitioning

#### 2. **Quantum Router** (`src/router/`)
- ML-based decision engine for solver selection
- Evaluates: problem size, hardware constraints, time budgets
- Routes to: Classical, Quantum, or Hybrid approaches
- Provides confidence scores and reasoning for decisions

#### 3. **Solver Ecosystem** (`src/solvers/`)

##### ✅ Implemented Solvers
- **Classical Solvers**: 
  - Greedy algorithms (MaxCut)
  - Simulated Annealing (MaxCut, TSP, generic)
  - OR-Tools (TSP with fallback)
  - SciPy optimizers (Portfolio: Sharpe ratio maximization, minimum variance)
- **Quantum Solvers**: 
  - QAOA (Quantum Approximate Optimization Algorithm)
  - VQE (Variational Quantum Eigensolver)
  - Photonic quantum simulation with noise modeling
- **Hybrid Solvers**: 
  - Adaptive strategy (problem-size based routing)
  - Quantum-assisted classical refinement
  - Classical-first with quantum enhancement
  - Parallel execution with best result selection
  - Iterative quantum-classical collaboration

##### 🚧 Planned/Not Implemented (Require Real Hardware or Licensing)
- **Classical Solvers**: 
  - Gurobi (requires commercial license)
  - NetworkX advanced optimizers
- **Quantum Hardware Interfaces**: 
  - IBM Quantum (requires IBM Quantum account)
  - AWS Braket (requires AWS account)
  - Rotonium QPU (requires Rotonium hardware access)
- **Quantum Algorithms**: 
  - Quantum Annealing on real hardware (D-Wave)

**Note**: Abstract interfaces for IBM Quantum, AWS Braket, and Rotonium are provided in `src/solvers/quantum_hardware_interface.py` but require actual SDK integration and hardware access credentials.

#### 4. **Monitoring System** (`src/monitoring/`)
- Tracks execution time, memory usage, energy consumption
- Compares solution quality across solvers
- Stores historical performance data in TimescaleDB
- Provides performance metrics and analytics

#### 5. **API Layer** (`src/api/`)
- RESTful FastAPI endpoints
- Async job submission and status tracking
- Comparative analysis endpoints
- OpenAPI/Swagger documentation

#### 6. **Target Profiles & Backends**
- **Profiles** (`profiles/*.yaml`) describe company branding, deployment options, energy models, demo scenarios, and docs references
- **Profile Loader** (`src/profile_loader.py`) resolves the active profile via `--profile` CLI flag or env var and validates it with Pydantic
- **Backend Interface** (`src/backends/`) implements the new `QuantumBackend` ABC
  - `RotoniumMockBackend`: Simulates photonic OAM hardware for edge deployments
  - `QuiXCloudBackend`: Connects to the real QuiX API (or mock mode when no key is supplied)
- **Datacenter Simulator** (`src/router/datacenter_simulator.py`) models HPC/rack/cloud resources and PUE-adjusted energy budgets for QuiX scenarios

### Data Flow Explanation

1. **Problem Submission** → User submits optimization problem via API or dashboard
2. **Analysis** → Problem Analyzer extracts features and metadata
3. **Routing Decision** → Quantum Router selects optimal solver(s)
4. **Execution** → Selected solver(s) process the problem
5. **Result Collection** → Metrics and solutions are stored in PostgreSQL
6. **Response** → Results returned to user with performance metrics

---

##  Usage Examples

For comprehensive usage examples including API calls, Python SDK integration, and direct solver usage, see **[docs/usage-examples.md](docs/usage-examples.md)**.

### Quick Examples

#### Submit a Problem via API
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/maxcut" \
  -H "Content-Type: application/json" \
  -d '{"num_nodes": 10, "edge_probability": 0.3, "strategy": "balanced"}'
```

#### Python SDK - Direct Solver Usage
```python
from src.problems.maxcut import MaxCutProblem
from src.solvers.quantum_simulator import QuantumSimulator

problem = MaxCutProblem(num_nodes=10)
problem.generate(edge_probability=0.5, seed=42)

solver = QuantumSimulator(backend='default.qubit', shots=1024)
result = solver.solve(problem, algorithm='qaoa')  # or 'vqe'
```

For more examples including:
- Comparative analysis between classical and quantum solvers
- System status and statistics queries
- Hybrid solver strategies
- Portfolio optimization
- Router-based automatic solver selection

See the complete guide: **[docs/usage-examples.md](docs/usage-examples.md)**

---

##  Rotonium Integration

### How This Supports Rotonium's Vision

QuantumEdge Pipeline showcase and accelerate **Rotonium's photonic quantum processors** in edge computing scenarios. Key alignment:

#### 1. **Photonic QPU Simulation**
- **Room Temperature Operation**: Simulation models that reflect photonic quantum computing advantages
- **OAM Encoding**: Conceptual framework for orbital angular momentum-based qubit encoding
- **Low Power Consumption**: Energy models optimized for photonic operations vs. cryogenic systems
- **Edge SWaP Optimization**: Designed for Size, Weight, and Power constrained environments

#### 2. **Edge Deployment Focus**
The pipeline demonstrates Rotonium's competitive advantage in:
- **Aerospace Applications**: Flight path optimization, satellite constellation routing
- **Defense & NATO DIANA**: Secure tactical optimization at the edge
- **Mobile Edge Computing**: 5G/6G network optimization with quantum acceleration
- **Space-Based Computing**: Radiation-resistant, low-power quantum operations

#### 3. **Path to Real Hardware Integration**
Clear integration pathway from simulation to real QPU (Abstract interfaces implemented, SDK integration required):

```python
# ✅ Current: Simulation (Fully Implemented)
from src.solvers.quantum_simulator import QuantumSimulator
solver = QuantumSimulator(backend='default.qubit', shots=1024)
result = solver.solve(problem, algorithm='qaoa')  # or 'vqe'

# 🚧 Future: Rotonium Hardware (Abstract Interface Ready)
from src.solvers.quantum_hardware_interface import create_hardware_interface
solver = create_hardware_interface(
    'rotonium',
    api_key='your_api_key',
    device='rotonium_photonic_qpu_v1'
)
# Note: Requires Rotonium SDK and hardware access credentials
result = solver.submit_job(problem)

# 🚧 Alternative: IBM Quantum (Abstract Interface Ready)
solver = create_hardware_interface(
    'ibm',
    api_key='your_ibm_token',
    backend='ibm_nairobi'
)
# Note: Requires qiskit-ibm-runtime package and IBM Quantum account
result = solver.submit_job(problem)

# 🚧 Alternative: AWS Braket (Abstract Interface Ready)
solver = create_hardware_interface(
    'aws',
    region='us-east-1',
    device='Rigetti/Aspen-M-3'
)
# Note: Requires AWS credentials and Braket SDK
result = solver.submit_job(problem)
```

**Integration Status**:
- ✅ Abstract interfaces defined in `src/solvers/quantum_hardware_interface.py`
- ✅ Standardized job submission and result retrieval methods
- 🚧 Requires actual SDK integration (qiskit-ibm-runtime, AWS Braket SDK, Rotonium SDK)
- 🚧 Requires hardware access credentials and accounts

#### 4. **Value Proposition for Customers**

| Benefit | Description |
|---------|-------------|
| **Reduced Integration Time** | Pre-built workflows and APIs ready for production |
| **Early Viability Proof** | Demonstrate quantum advantage before hardware deployment |
| **Developer-Friendly Tools** | Familiar REST APIs, Python SDKs, Docker deployment |
| **Ecosystem Readiness** | Compatible with existing quantum frameworks (PennyLane, qiskit-ibm-runtime) |
| **Competitive Benchmarking** | Side-by-side comparison with IBM, Rigetti, D-Wave |

### Competitive Advantages: Photonic vs. Cryogenic

| Aspect | Rotonium (Photonic) | Cryogenic Systems |
|--------|---------------------|-------------------|
| **Operating Temperature** | Room temperature (20°C) | Near absolute zero (~0.01K) |
| **Power Consumption** | <100W | 10-25 kW (incl. cooling) |
| **Size (SWaP)** | Rack-mountable (2U-4U) | Building-sized infrastructure |
| **Deployment Speed** | Minutes | Days to weeks |
| **Edge Suitability** | Excellent | Impractical |
| **Maintenance** | Minimal | Complex cryogenic systems |

For detailed integration guide, see [**docs/rotonium-integration.md**](docs/rotonium-integration.md)

---

## QuiX Quantum Integration

The Target Profiles architecture also ships with a full QuiX configuration focused on HPC/data-center deployments. Highlights:

- **QuiXCloudBackend** connects to `cloud.quixquantum.com` with `httpx` when `QUIX_API_KEY` is provided, or uses a high-fidelity mock when testing locally.
- **DatacenterEnvironment** models `hpc_cluster`, `datacenter_rack`, and `cloud_node` profiles with realistic power, GPU, and bandwidth constraints plus PUE-based energy accounting.
- **Demo Scenarios** (pharma optimization, portfolio risk, hydrology) showcase how silicon-nitride photonic processors slot into HPC pipelines.
- **Dashboard & API** automatically adapt branding, metrics, and documentation links when `QUANTUMEDGE_PROFILE=quix`.

For the full guide, see [**docs/quix-integration.md**](docs/quix-integration.md).

---

##  Demo Scenarios

Run these demos on the interactive dashboard at http://localhost:8501

### Using Demo Scenarios in the Dashboard

The Streamlit dashboard includes pre-configured demo scenarios that automatically populate the problem submission form:

1. **Navigate to "Submit Problem" page**
2. **Select a demo scenario** from the dropdown (e.g., "AEROSPACE_ROUTING", "FINANCIAL_PORTFOLIO", "ML_GRAPH_PARTITION",..)
3. **Click "Load Demo"** - Form fields will be populated with demo data
4. **Optional:** Modify any parameters as needed (form will remember your changes)
5. **Click "🚀 Submit Job"** to run the problem
6. **Click "Clear Demo"** to reset to default values


### Rotonium Scenarios (Edge)

#### 1.  Aerospace Routing Optimization

**Scenario**: Optimize flight paths for a fleet of drones performing surveillance over a region.

**Demo Configuration**: `AEROSPACE_ROUTING`
- **Problem Type**: MaxCut (40 nodes)
- **Edge Profile**: Aerospace (50W power budget)
- **Strategy**: Energy-optimized
- **Comparative Mode**: Enabled

**Problem Details**:
- 40 waypoints representing drone coordination zones
- 25% edge probability (sparse graph for efficient routing)
- Power constraints (battery-powered UAVs)
- Quantum advantage demonstrated for energy-constrained scenarios

**Expected Results**:
- Classical solver: Fast but higher power consumption
- Quantum QAOA: Energy-efficient solution within power budget
- Winner: Quantum (energy strategy prioritizes lower consumption)

#### 2.  Financial Portfolio Optimization

**Scenario**: Asset allocation for risk-constrained portfolio with correlation matrix.

**Demo Configuration**: `FINANCIAL_PORTFOLIO`
- **Problem Type**: Portfolio (50 assets)
- **Edge Profile**: Ground Server (200W power budget)
- **Strategy**: Quality-optimized
- **Comparative Mode**: Enabled

**Problem Details**:
- 50 assets with historical returns
- Risk constraints (max variance, sector limits)
- Correlation-aware optimization
- Real-time market data integration

**Expected Results**:
- Quantum solver finds higher quality portfolios
- Ground server profile provides ample power for quantum exploration
- Winner: Quantum (quality strategy prioritizes solution superiority)

#### 3.  ML Graph Partitioning

**Scenario**: Partition neural network graph for distributed training across edge devices.

**Demo Configuration**: `ML_GRAPH_PARTITION`
- **Problem Type**: MaxCut (60 nodes)
- **Edge Profile**: Mobile (5-15W power budget)
- **Strategy**: Balanced
- **Comparative Mode**: Enabled

**Problem Details**:
- 60 nodes with 40% density (interconnected layers)
- Minimize inter-device communication
- Balance compute load across mobile edge nodes
- Challenging for both classical and quantum approaches

**Expected Results**:
- Balanced strategy weighs time, energy, and quality equally
- Mobile power constraints are restrictive
- Winner: Context-dependent (varies based on runtime analysis)

### QuiX Scenarios (Data Center)

#### 4. Drug Discovery Sampling
- **Profile**: `PHARMA_OPTIMIZATION`
- **Problem Type**: MaxCut (50 nodes)
- **Deployment**: HPC Cluster (PUE 1.2)
- **Strategy**: Quality-optimized
- **Highlights**: Demonstrates photonic sampling advantages for molecular conformation search.

#### 5. Financial Risk Modeling (Datacenter)
- **Profile**: `PORTFOLIO_RISK`
- **Problem Type**: Portfolio (40 assets)
- **Deployment**: Datacenter rack
- **Strategy**: Balanced
- **Highlights**: Quantum sampling benchmarks for VaR/ES calculations alongside classical Monte Carlo.

#### 6. Hydrology Simulation (Cloud)
- **Profile**: `HYDROLOGY`
- **Problem Type**: MaxCut (45 nodes)
- **Deployment**: Cloud node
- **Strategy**: Balanced
- **Highlights**: Cloud-hosted QuiX hardware models water-network optimization with PUE-adjusted energy metrics.

---

##  Development

### Project Structure

```
quantumedge-pipeline/
├── profiles/                    # Company profiles (rotonium, quix, default)
├── src/                          # Main source code
│   ├── __init__.py
│   ├── config.py                 # Global configuration settings
│   ├── profile_loader.py         # Loads/validates company profiles
│   ├── analyzer/                 # Problem analysis module
│   │   ├── __init__.py
│   │   └── problem_analyzer.py   # Problem characterization & feature extraction
│   ├── api/                      # REST API layer
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application entry point
│   │   └── orchestrator.py       # Job execution orchestrator
│   ├── monitoring/               # Performance monitoring & metrics
│   │   ├── __init__.py
│   │   ├── metrics_collector.py  # Performance tracking & metrics
│   │   └── db_manager.py         # Database operations & persistence
│   ├── problems/                 # Problem type implementations
│   │   ├── __init__.py
│   │   ├── problem_base.py       # Abstract problem interface
│   │   ├── maxcut.py             # Maximum Cut problem
│   │   ├── tsp.py                # Traveling Salesman Problem
│   │   └── portfolio.py          # Portfolio optimization
│   ├── router/                   # Intelligent routing engine
│   │   ├── __init__.py
│   │   ├── quantum_router.py     # Solver selection & routing logic
│   │   ├── edge_simulator.py     # Edge environment simulation
│   │   └── datacenter_simulator.py # HPC/data-center environment simulation
│   └── solvers/                  # Solver implementations
│       ├── __init__.py
│       ├── solver_base.py        # Abstract solver interface
│       ├── classical_solver.py   # Classical optimization solvers
│       └── quantum_simulator.py  # Quantum algorithm simulators
│   └── backends/                 # Pluggable quantum hardware backends
│       ├── backend_base.py       # QuantumBackend ABC
│       ├── rotonium_mock.py      # Photonic mock backend
│       └── quix_cloud.py         # QuiX cloud API client (real/mock)
│
├── examples/
│   └── scenarios/                # JSON payloads for dashboard demo scenarios
│
├── dashboard/                    # Streamlit web dashboard
│   ├── app.py                    # Main dashboard application
│   ├── demo_scenarios.py         # Demo scenario configurations
│   └── utils.py                  # Visualization & utility functions
│
├── scripts/                      # Utility & setup scripts
│   ├── benchmark_solvers.py      # Solver performance benchmarking
│   └── init_db.sql               # Database initialization SQL
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_analyzer.py          # Problem analyzer tests
│   ├── test_api.py               # API endpoint tests
│   ├── test_monitoring.py        # Monitoring & metrics tests
│   ├── test_problems.py          # Problem implementation tests
│   ├── test_router.py            # Routing logic tests
│   └── test_solvers.py           # Solver implementation tests
│
├── docs/                         # Documentation
│   ├── api.md                    # API reference documentation
│   ├── docker-services.md        # Docker services overview
│   ├── quantum-basics.md         # Quantum computing primer
│   ├── quickstart.md             # Quick start guide
│   └── rotonium-integration.md   # Rotonium integration guide
│
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Multi-service orchestration
├── Makefile                      # Development & deployment commands
├── pyproject.toml                # Poetry project configuration
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .dockerignore                 # Docker build exclusions
├── .gitignore                    # Git ignore patterns
└── README.md                     # This file
```

### Running Tests

```bash
# Run all unit tests with coverage
make test

# Run specific test suite
docker-compose exec api poetry run pytest tests/test_router.py -v

# Run tests with coverage report
make test-cov
```

### Running Benchmarks

The project includes a standalone benchmark script for comprehensive performance analysis of classical vs quantum solvers. 
This is **not** part of the pytest suite but a separate utility for detailed solver comparison.

```bash
# Run benchmark with default settings
docker-compose exec api python scripts/benchmark_solvers.py

# Run benchmark with custom problem sizes
docker-compose exec api python scripts/benchmark_solvers.py --sizes 10 20 30 40 50

# Run benchmark and generate performance plots
docker-compose exec api python scripts/benchmark_solvers.py --plot

# Run benchmark with more repetitions for statistical significance
docker-compose exec api python scripts/benchmark_solvers.py --repetitions 5

# Run benchmark without database storage
docker-compose exec api python scripts/benchmark_solvers.py --no-db

# Analyze existing benchmark results
docker-compose exec api python scripts/benchmark_solvers.py --csv benchmark_results/benchmark_20240115_143022.csv --plot
```

**Benchmark Features:**
- Compares Classical (Greedy, Simulated Annealing) vs Quantum (QAOA) solvers
- Measures execution time, energy consumption, and solution quality
- Identifies problem sizes where quantum advantage emerges
- Generates CSV reports, visualizations, and markdown summaries
- Results stored in `benchmark_results/` directory

**Output Files:**
- `benchmark_YYYYMMDD_HHMMSS.csv` - Raw benchmark data
- `benchmark_report_YYYYMMDD_HHMMSS.md` - Analysis report with recommendations
- `performance_plots_YYYYMMDD_HHMMSS.png` - Visualization charts (if --plot used)

---
##  Technical Details

### Solver Algorithms Used

#### ✅ Implemented Classical Solvers
- **Greedy Algorithms**: Fast approximation for MaxCut problems (O(n×m) complexity)
- **Simulated Annealing**: Metaheuristic for MaxCut, TSP, and generic optimization
- **OR-Tools**: Constraint programming for TSP with fallback to simulated annealing
- **SciPy**: Continuous optimization (SLSQP) for portfolio optimization
  - Sharpe ratio maximization
  - Minimum variance portfolio with target return constraints
  - Constraint handling (weights sum to 1, no short selling)

#### ✅ Implemented Quantum Solvers
- **QAOA (Quantum Approximate Optimization Algorithm)**:
  - Variational quantum algorithm for combinatorial optimization
  - Parameterized quantum circuits with classical optimization loop
  - Supports: COBYLA, BFGS, L-BFGS-B, Nelder-Mead optimizers
  - Configurable circuit depth (p layers)
  - Best for: MaxCut, Graph Coloring, Number Partitioning
  
- **VQE (Variational Quantum Eigensolver)**:
  - Hybrid quantum-classical approach for ground state problems
  - Hardware-efficient ansatz with configurable layer depth
  - Same classical optimizer framework as QAOA
  - Best for: Portfolio optimization, Molecular simulation
  
- **Photonic Quantum Simulation**:
  - Room-temperature operation model
  - Photonic noise characteristics (photon loss, detection efficiency)
  - OAM (Orbital Angular Momentum) encoding support
  - Energy efficiency modeling (~10,000x vs cryogenic systems)

#### ✅ Implemented Hybrid Solvers
- **Adaptive Strategy**: Automatic routing based on problem size and complexity
- **Quantum-Assisted**: Classical preprocessing with quantum refinement
- **Classical-First**: Classical solution with quantum enhancement attempts
- **Parallel Execution**: Run both solvers and select best result
- **Iterative Collaboration**: Multi-round quantum-classical optimization

#### 🚧 Planned (Not Implemented - Require External Resources)
- **Gurobi**: Mixed Integer Linear Programming (MILP) - requires commercial license
- **NetworkX**: Advanced graph algorithms - integration planned
- **Quantum Annealing**: Adiabatic quantum computation on real hardware (D-Wave)
- **IBM Quantum**: Real quantum hardware execution (requires IBM Quantum account)
- **AWS Braket**: Cloud quantum computing service (requires AWS account)
- **Rotonium QPU**: Photonic quantum processor (requires hardware access)

### Routing Decision Logic

The Quantum Router uses a multi-criteria decision framework:

```python
def routing_score(problem, solver):
    score = (
        0.4 * performance_score(problem, solver) +
        0.3 * resource_efficiency(problem, solver) +
        0.2 * solution_quality(problem, solver) +
        0.1 * energy_efficiency(problem, solver)
    )
    return score
```

**Decision Factors**:
1. **Problem Size**: <10 qubits → Quantum, >50 qubits → Classical
2. **Graph Structure**: High connectivity → Classical, Sparse → Quantum
3. **Time Constraints**: Strict time limits favor classical approximations
4. **Hardware Availability**: Route to available quantum backend or simulate

### Energy Measurement Approach

Energy consumption tracking for fair solver comparison:

```python
# Classical solver energy model
energy_classical = (
    cpu_power_watts * execution_time_seconds +
    memory_power_watts * peak_memory_gb * execution_time_seconds
)

# Quantum solver energy model (simulation)
energy_quantum = (
    gate_energy_per_qubit * num_qubits * circuit_depth +
    classical_overhead_energy  # Parameter optimization
)

# Rotonium photonic QPU (estimated)
energy_rotonium = (
    laser_power_watts * execution_time_seconds +
    detector_power_watts * num_measurements
)
# Typically 10-100x more efficient than cryogenic systems
```

### Database Schema

For detailed schema and indexing strategy, see `database/init.sql`.

**Important Note**: This is a research and development platform designed for evaluation, benchmarking, and integration planning. 
