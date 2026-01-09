# Usage Examples

This document provides comprehensive examples for using the QuantumEdge Pipeline through various interfaces: REST API, Python SDK, and direct solver integration.

## Table of Contents

1. [Submit a Problem via API](#1-submit-a-problem-via-api)
2. [Run Comparative Analysis](#2-run-comparative-analysis)
3. [Query System Status and Statistics](#3-query-system-status-and-statistics)
4. [Python SDK Examples](#4-python-sdk-examples)

---

## 1. Submit a Problem via API

Submit a MaxCut optimization problem through the REST API:

### Python Example

```python
import httpx
import asyncio

async def submit_maxcut_problem():
    """
    Alternative MaxCut test script that matches the actual API implementation.
    This script uses the correct endpoint and request schema.
    """
    async with httpx.AsyncClient() as client:
        try:
            # Using the correct endpoint and schema
            response = await client.post(
                "http://localhost:8000/api/v1/jobs/maxcut",
                json={
                    "num_nodes": 10,
                    "edge_probability": 0.3,
                    "edge_profile": "aerospace",
                    "strategy": "balanced",
                    "seed": 42
                },
                timeout=30.0
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            result = response.json()
            
            # Display results
            print("=" * 60)
            print("MaxCut Job Results")
            print("=" * 60)
            print(f"Job ID: {result['job_id']}")
            print(f"Solver Used: {result.get('solver_used', 'N/A')}")
            print(f"Solution: {result.get('solution', 'N/A')}")
            
            return result
            
        except httpx.ConnectError as e:
            print("Connection Error: Could not connect to the API server.")
            print(f"Error: {e}")
            return None
            
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            return None
            
        except Exception as e:
            print(f"Unexpected Error: {type(e).__name__}")
            print(f"Error: {e}")
            return None

# Run the async function
if __name__ == "__main__":
    result = asyncio.run(submit_maxcut_problem())
    
    if result:
        print("Test completed successfully!")
    else:
        print("Test failed - see errors above")

```

### cURL Equivalent

```bash
curl -s -w "\n%{http_code}" -X POST "http://localhost:8000/api/v1/jobs/maxcut" \
  -H "Content-Type: application/json" \
  -d '{
    "num_nodes": 10,
    "edge_probability": 0.3,
    "edge_profile": "aerospace",
    "strategy": "balanced",
    "seed": 42
  }'
```

---

## 2. Run Comparative Analysis

Compare classical vs. quantum solvers side-by-side:

```python
import httpx
import asyncio

async def compare_solvers():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/jobs/comparative",
            json={
                "problem_type": "tsp",
                "problem_size": 5,  # Number of cities (changed from cities array)
                "edge_profile": "aerospace",  # Optional: aerospace, mobile, or ground_server
                "seed": 42  # Optional: for reproducibility
            },
            timeout=120.0
        )
        comparison = response.json()
        
        print("=== Solver Comparison ===")
        
        # Classical solver results
        print("\n[Classical Solver]")
        print(f"  Time: {comparison['classical']['time_ms']:.3f}ms")
        print(f"  Cost: {comparison['classical']['cost']}")
        print(f"  Energy: {comparison['classical']['energy_mj']:.2f}mJ")
        print(f"  Valid: {comparison['classical']['is_valid']}")
        print(f"  Quality: {comparison['classical']['solution_quality']:.2%}")
        if comparison['classical'].get('error'):
            print(f"  Error: {comparison['classical']['error']}")
        
        # Quantum solver results
        print("\n[Quantum Solver]")
        print(f"  Time: {comparison['quantum']['time_ms']:.3f}ms")
        print(f"  Cost: {comparison['quantum']['cost']}")
        print(f"  Energy: {comparison['quantum']['energy_mj']:.2f}mJ")
        print(f"  Valid: {comparison['quantum']['is_valid']}")
        print(f"  Quality: {comparison['quantum']['solution_quality']:.2%}")
        if comparison['quantum'].get('error'):
            print(f"  Error: {comparison['quantum']['error']}")
        
        # Comparison metrics
        print("\n[Comparison Metrics]")
        if comparison.get('speedup_factor'):
            print(f"  Speedup Factor: {comparison['speedup_factor']:.2f}x")
        if comparison.get('energy_ratio'):
            print(f"  Energy Ratio: {comparison['energy_ratio']:.2f}x")
        if comparison.get('quality_diff'):
            print(f"  Quality Difference: {comparison['quality_diff']:.2%}")
        
        print(f"\n[Recommendation]")
        print(f"  Winner: {comparison['recommendation']}")
        print(f"  Reason: {comparison['recommendation_reason']}")
        
        return comparison

asyncio.run(compare_solvers())
```

---

## 3. Query System Status and Statistics

Retrieve system health, configuration, and execution statistics:

```python
import httpx
import asyncio

async def test_system_and_execute():
    """
    Working alternative that demonstrates available functionality.
    Since job retrieval and historical metrics endpoints don't exist,
    we'll submit a job and immediately get results, then check system stats.
    """
    async with httpx.AsyncClient() as client:
        
        # 1. Check system health
        print("=== System Health Check ===")
        response = await client.get("http://localhost:8000/health")
        health = response.json()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Components: {health['components']}")
        
        # 2. Get system information
        print("\n=== System Information ===")
        response = await client.get("http://localhost:8000/api/v1/system/info")
        info = response.json()
        print(f"Application: {info['application']['name']}")
        print(f"Environment: {info['application']['environment']}")
        print(f"Quantum Backend: {info['configuration']['quantum_backend']}")
        print(f"Max Qubits: {info['configuration']['quantum_max_qubits']}")
        print(f"Problem Types: {info['capabilities']['problem_types']}")
        print(f"Solver Types: {info['capabilities']['solver_types']}")
        
        # 3. Submit a MaxCut job and get immediate results
        print("\n=== Submitting MaxCut Job ===")
        response = await client.post(
            "http://localhost:8000/api/v1/jobs/maxcut",
            json={
                "num_nodes": 20,
                "edge_probability": 0.3,
                "edge_profile": "aerospace",
                "strategy": "balanced",
                "seed": 42
            },
            timeout=60.0
        )
        job_result = response.json()
        print(F"Debug: {job_result.keys()}")
        print(f"Job ID: {job_result['job_id']}")
        print(f"Status: {job_result['status']}")
        print(f"Problem Size: {job_result['problem_size']}")
        print(f"Solver Used: {job_result['solver_used']}")
        print(f"Routing Decision: {job_result['routing_decision']}")
        print(f"Routing Confidence: {job_result['routing_confidence']:.2%}")
        print(f"Cost: {job_result['cost']}")
        print(f"Execution Time: {job_result['time_ms']:.2f}ms")
        print(f"Energy Consumed: {0 if job_result['energy_consumed_mj'] is None else job_result['energy_consumed_mj']:.2f}mJ")
        print(f"Solution Valid: {job_result['is_valid']}")
        print(f"Solution Quality: {job_result['solution_quality']:.2%}")
        print(f"Reasoning: {job_result['routing_reason']}")
        
        # 4. Get system statistics (in-memory only)
        print("\n=== System Statistics ===")
        response = await client.get("http://localhost:8000/api/v1/system/stats")
        stats = response.json()
        print(f"Jobs Executed: {stats['statistics']['jobs_executed']}")
        print(f"Jobs Failed: {stats['statistics']['jobs_failed']}")
        print(f"Success Rate: {stats['statistics']['success_rate']:.2%}")
        print(f"Avg Execution Time: {stats['statistics']['average_execution_time_ms']:.2f}ms")
        print(f"Total Execution Time: {stats['statistics']['total_execution_time_ms']:.2f}ms")
        
        # 5. Submit comparative analysis
        print("\n=== Comparative Analysis ===")
        response = await client.post(
            "http://localhost:8000/api/v1/jobs/comparative",
            json={
                "problem_type": "tsp",
                "problem_size": 10,
                "edge_profile": "aerospace",
                "seed": 42
            },
            timeout=120.0
        )
        comparison = response.json()
        
        print(f"\nClassical Solver:")
        print(f"  Time: {comparison['classical']['time_ms']:.2f}ms")
        print(f"  Cost: {comparison['classical']['cost']:.4f}")
        print(f"  Quality: {comparison['classical']['solution_quality']:.2%}")
        
        print(f"\nQuantum Solver:")
        print(f"  Time: {comparison['quantum']['time_ms']:.2f}ms")
        print(f"  Cost: {comparison['quantum']['cost']:.4f}")
        print(f"  Quality: {comparison['quantum']['solution_quality']:.2%}")
        
        print(f"\nRecommendation: {comparison['recommendation']}")
        print(f"Reason: {comparison['recommendation_reason']}")
        
        # 6. Check edge profiles
        print("\n=== Edge Profiles ===")
        response = await client.get("http://localhost:8000/api/v1/config/edge-profiles")
        profiles = response.json()
        for name, profile in profiles['profiles'].items():
            print(f"\n{name.upper()}:")
            print(f"  Power Budget: {profile['power_budget_watts']}W")
            print(f"  Memory: {profile['memory_mb']}MB")
            print(f"  CPU Cores: {profile['cpu_cores']}")
            print(f"  Max Execution Time: {profile['max_execution_time_sec']}s")

asyncio.run(test_system_and_execute())
```

---

## 4. Python SDK Examples

For direct integration without API calls, use the Python SDK:

### Example 1: Using Router for Automatic Solver Selection

```python
from src.problems.maxcut import MaxCutProblem
from src.router.quantum_router import QuantumRouter, RoutingStrategy
from src.router.edge_simulator import EdgeEnvironment, DeploymentProfile
from src.solvers.classical_solver import ClassicalSolver
from src.solvers.quantum_simulator import QuantumSimulator

# Create and generate problem
problem = MaxCutProblem(num_nodes=15)
problem.generate(edge_probability=0.3, seed=42)

# Create router with balanced strategy
router = QuantumRouter(strategy=RoutingStrategy.BALANCED, enable_learning=True)
edge_env = EdgeEnvironment(DeploymentProfile.AEROSPACE)

# Route problem to appropriate solver
routing_result = router.route_problem(problem, edge_env)

# Get and execute solver
decision = routing_result['decision']
if decision == 'classical':
    solver = ClassicalSolver(default_method='simulated_annealing')
    print("Using ClassicalSolver (simulated annealing)")
elif decision == 'quantum':
    solver = QuantumSimulator(shots=1024)
    print("Using QuantumSimulator (QAOA)")
else:
    solver = ClassicalSolver()

result = solver.solve(problem)
print(f"Solution Quality: {problem.validate_solution(result['solution'])}")
```

### Example 2: Using Quantum Simulator Directly with VQE

```python
from src.problems.maxcut import MaxCutProblem
from src.solvers.quantum_simulator import QuantumSimulator

# Create problem
problem = MaxCutProblem(num_nodes=10)
problem.generate(edge_probability=0.5, seed=42)

# Use quantum simulator with VQE algorithm
quantum_solver = QuantumSimulator(backend='default.qubit', shots=1024)
result = quantum_solver.solve(problem, algorithm='vqe', ansatz_depth=3, maxiter=100)

print(f"VQE Result: Cost={result['cost']}, Time={result['time_ms']}ms")
```

### Example 3: Using Hybrid Solver with Adaptive Strategy

```python
from src.problems.maxcut import MaxCutProblem
from src.solvers.hybrid_solver import HybridSolver, HybridStrategy

# Create problem
problem = MaxCutProblem(num_nodes=20)
problem.generate(edge_probability=0.3, seed=42)

# Use hybrid solver with adaptive strategy
hybrid_solver = HybridSolver(strategy=HybridStrategy.ADAPTIVE)
result = hybrid_solver.solve(problem)

print(f"Hybrid Strategy: {result['metadata']['strategy_used']}")
print(f"Execution Path: {result['metadata']['execution_path']}")
```

### Example 4: Portfolio Optimization with SciPy

```python
from src.problems.portfolio import PortfolioProblem
from src.solvers.classical_solver import ClassicalSolver

# Create portfolio problem
portfolio = PortfolioProblem(num_assets=20, num_selected=5)
portfolio.generate(seed=42)

# Solve using SciPy optimizer
classical_solver = ClassicalSolver(default_method='scipy')
result = classical_solver.solve(portfolio, method='sharpe')

print(f"Portfolio Sharpe Ratio: {-result['cost']:.4f}")
print(f"Selected Assets: {result['solution']}")
```

---

## Additional Resources

- **API Documentation**: http://localhost:8000/docs (when services are running)
- **Quick Start Guide**: [docs/quickstart.md](quickstart.md)
- **API Reference**: [docs/api.md](api.md)
- **Main README**: [README.md](../README.md)
