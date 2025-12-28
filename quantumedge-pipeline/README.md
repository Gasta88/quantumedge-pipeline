# QuantumEdge Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A quantum-classical hybrid optimization pipeline designed for edge computing environments. This system intelligently routes computational problems between classical and quantum solvers based on problem characteristics, resource constraints, and performance requirements.

## 🚀 Features

- **Intelligent Problem Analysis**: Automatic characterization of optimization problems
- **Smart Routing**: Dynamic decision-making for classical vs. quantum execution
- **Edge-Optimized**: Designed for resource-constrained edge environments
- **Multiple Problem Types**: MaxCut, TSP, Portfolio Optimization, and extensible architecture
- **Quantum Simulation**: Built-in quantum circuit simulation with Qiskit and PennyLane
- **Real-time Monitoring**: Performance metrics and execution tracking
- **REST API**: FastAPI-based interface for integration
- **Interactive Dashboard**: Streamlit-based visualization and control
- **Docker Support**: Complete containerized deployment

## 📋 Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- Docker & Docker Compose (optional, for containerized deployment)
- PostgreSQL 16+ (or use Docker)
- Redis 7+ (or use Docker)

## 🔧 Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd quantumedge-pipeline

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd quantumedge-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Using Docker

```bash
# Clone the repository
git clone <repository-url>
cd quantumedge-pipeline

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

## 🏃 Quick Start

### Initialize Database

```bash
# Using Docker
docker-compose exec postgres psql -U quantumedge_user -d quantumedge -f /docker-entrypoint-initdb.d/init.sql

# Or locally
python scripts/seed_data.py
```

### Start API Server

```bash
# Using Poetry
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Using Python directly
python -m uvicorn src.api.main:app --reload
```

Visit API documentation at: http://localhost:8000/docs

### Start Dashboard

```bash
# Using Poetry
poetry run streamlit run dashboard/app.py

# Using Python directly
streamlit run dashboard/app.py
```

Visit dashboard at: http://localhost:8501

### Run Demo

```bash
# Using Poetry
poetry run python scripts/run_demo.py

# Using Python directly
python scripts/run_demo.py
```

## 📁 Project Structure

```
quantumedge-pipeline/
├── src/                    # Main source code
│   ├── analyzer/           # Problem characterization
│   ├── router/             # Routing logic
│   ├── solvers/            # Classical & quantum solvers
│   ├── monitoring/         # Metrics & monitoring
│   ├── problems/           # Problem implementations
│   └── api/                # REST API
├── dashboard/              # Streamlit dashboard
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
└── examples/               # Example notebooks & scenarios
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_router.py

# Run with verbose output
poetry run pytest -v
```

## 🎯 Usage Examples

### API Example

```python
import httpx

# Submit optimization problem
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/optimize",
        json={
            "problem_type": "maxcut",
            "graph": {"nodes": 10, "edges": [[0, 1], [1, 2], ...]},
            "constraints": {"max_time": 60, "max_memory": 2048}
        }
    )
    result = response.json()
    print(f"Solution: {result['solution']}")
    print(f"Solver used: {result['solver_type']}")
```

### Python SDK Example

```python
from src.router.quantum_router import QuantumRouter
from src.problems.maxcut import MaxCutProblem

# Create problem instance
problem = MaxCutProblem(num_nodes=12)

# Initialize router
router = QuantumRouter()

# Get routing decision
decision = router.route(problem)
print(f"Recommended solver: {decision.solver_type}")
print(f"Confidence: {decision.confidence}")

# Execute with recommended solver
result = decision.solver.solve(problem)
print(f"Objective value: {result.objective_value}")
```

## 🔬 Quantum Computing Integration

This pipeline supports multiple quantum computing backends:

- **Qiskit Aer Simulator** (default)
- **PennyLane** (for hybrid quantum-classical optimization)
- **IBM Quantum** (requires API token)
- **AWS Braket** (requires AWS credentials)

Configure your quantum backend in `.env`:

```bash
QUANTUM_BACKEND=simulator  # or ibm_quantum, aws_braket
QUANTUM_SHOTS=1000
MAX_QUBITS=20
```

## 📊 Monitoring & Metrics

The system collects comprehensive metrics:

- Problem characteristics (size, complexity, structure)
- Routing decisions and confidence scores
- Solver performance (execution time, memory usage)
- Solution quality (objective value, optimality gap)
- System resource utilization

Access metrics via:
- Prometheus: http://localhost:9090
- Dashboard: http://localhost:8501
- API: http://localhost:8000/metrics

## 🛠️ Development

### Code Formatting

```bash
# Format code with Black
poetry run black src/ tests/

# Lint with Ruff
poetry run ruff check src/ tests/

# Type checking with MyPy
poetry run mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Rotonium Integration Guide](docs/rotonium-integration.md)
- [Quantum Computing Basics](docs/quantum-basics.md)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Qiskit team for quantum computing framework
- PennyLane team for quantum machine learning tools
- FastAPI and Streamlit for excellent web frameworks

## 📧 Contact

For questions and support, please open an issue on GitHub.

---

**Note**: This is a research and educational project. For production quantum computing workloads, please consult with quantum computing experts and use appropriate hardware.
