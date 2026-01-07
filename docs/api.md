# 🔌 QuantumEdge Pipeline API Documentation

## Table of Contents
1. [Overview](#overview)
2. [API Endpoints](#api-endpoints)
3. [Request/Response Schemas](#requestresponse-schemas)
4. [Error Handling](#error-handling)
5. [Code Examples](#code-examples)
6. [Testing](#testing)

---

## Overview

The QuantumEdge Pipeline exposes a RESTful API built with **FastAPI** that allows developers to:
- Submit optimization problems (MaxCut, TSP, Portfolio) for quantum-classical hybrid solving
- Run comparative analyses between classical and quantum solvers
- Analyze routing decisions without execution
- Access edge profiles and routing strategies configuration
- Monitor system health and performance

**Base URL**: `http://localhost:8000` (development)

**API Version**: `v1`

**Supported Formats**: JSON

**Interactive Documentation**: 
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Endpoints

### System Endpoints

#### `GET /health`

Health check endpoint for monitoring service availability.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-01-06T20:00:00",
  "database": "disconnected"
}
```

#### `GET /api/v1/system/info`

Get comprehensive system information including configuration and capabilities.

**Response:**
```json
{
  "application": {
    "name": "QuantumEdge Pipeline API",
    "version": "0.1.0",
    "environment": "development"
  },
  "configuration": {
    "quantum_backend": "simulator",
    "edge_profiles": ["aerospace", "mobile", "ground"],
    "routing_strategies": ["balanced", "energy_optimized", "latency_optimized", "quality_optimized"]
  },
  "capabilities": {
    "problem_types": ["maxcut", "tsp", "portfolio"],
    "solver_types": ["classical", "quantum", "hybrid"],
    "max_problem_size": 100,
    "comparative_analysis": true
  }
}
```

#### `GET /api/v1/system/stats`

Get system statistics and performance metrics.

**Response:**
```json
{
  "total_jobs": 0,
  "successful_jobs": 0,
  "failed_jobs": 0,
  "average_execution_time_ms": 0.0,
  "solver_usage": {
    "classical": 0,
    "quantum": 0,
    "hybrid": 0
  },
  "uptime_seconds": 12345.67
}
```

---

### Configuration Endpoints

#### `GET /api/v1/config/edge-profiles`

Get available edge deployment profiles with resource constraints.

**Response:**
```json
{
  "profiles": {
    "aerospace": {
      "profile_name": "aerospace",
      "power_budget_watts": 50.0,
      "thermal_limit_celsius": 70.0,
      "memory_mb": 2048,
      "cpu_cores": 2,
      "max_execution_time_sec": 10,
      "network_latency_ms": 500
    },
    "mobile": {
      "profile_name": "mobile",
      "power_budget_watts": 15.0,
      "thermal_limit_celsius": 45.0,
      "memory_mb": 1024,
      "cpu_cores": 4,
      "max_execution_time_sec": 5,
      "network_latency_ms": 100
    },
    "ground": {
      "profile_name": "ground",
      "power_budget_watts": 200.0,
      "thermal_limit_celsius": 85.0,
      "memory_mb": 8192,
      "cpu_cores": 8,
      "max_execution_time_sec": 60,
      "network_latency_ms": 10
    }
  },
  "default_profile": "ground"
}
```

#### `GET /api/v1/config/routing-strategies`

Get available routing strategies and their descriptions.

**Response:**
```json
{
  "strategies": {
    "balanced": {
      "name": "balanced",
      "description": "Balance between speed, quality, and energy consumption",
      "use_case": "General-purpose optimization"
    },
    "energy_optimized": {
      "name": "energy_optimized",
      "description": "Minimize energy consumption",
      "use_case": "Battery-powered edge devices"
    },
    "latency_optimized": {
      "name": "latency_optimized",
      "description": "Minimize execution time",
      "use_case": "Real-time applications"
    },
    "quality_optimized": {
      "name": "quality_optimized",
      "description": "Maximize solution quality",
      "use_case": "Mission-critical optimization"
    }
  },
  "default_strategy": "balanced"
}
```

---

### Job Submission Endpoints

#### `POST /api/v1/jobs/maxcut`

Submit a MaxCut problem for solving.

**Request Body:**
```json
{
  "num_nodes": 30,
  "edge_probability": 0.3,
  "edge_profile": "aerospace",
  "strategy": "balanced",
  "seed": 42
}
```

**Parameters:**
- `num_nodes` (int, required): Number of nodes in the graph (5-100)
- `edge_probability` (float, optional): Probability of edge existence (0.0-1.0), default: 0.3
- `edge_profile` (str, optional): Edge deployment profile (aerospace/mobile/ground_server), default: "aerospace"
- `strategy` (str, optional): Routing strategy, default: "balanced"
- `seed` (int, optional): Random seed for reproducibility

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "problem_type": "maxcut",
  "problem_size": 30,
  "solver_used": "classical",
  "routing_decision": "classical",
  "routing_reason": "Quantum solver exceeds timeout constraints",
  "routing_confidence": 1.0,
  "solution": [1, 0, 1, 0, 1, 0, ...],
  "cost": -234.56,
  "time_ms": 12.5,
  "energy_consumed_mj": 0.625,
  "is_valid": true,
  "solution_quality": 0.95,
  "timestamp": "2026-01-06T20:00:00"
}
```

#### `POST /api/v1/jobs/tsp`

Submit a Traveling Salesman Problem (TSP) for solving.

**Request Body:**
```json
{
  "num_cities": 20,
  "euclidean": true,
  "edge_profile": "mobile",
  "strategy": "latency_optimized",
  "seed": 42
}
```

**Parameters:**
- `num_cities` (int, required): Number of cities (5-100)
- `euclidean` (bool, optional): Use Euclidean distance, default: true
- `edge_profile` (str, optional): Edge deployment profile
- `strategy` (str, optional): Routing strategy
- `seed` (int, optional): Random seed

**Response:** Same structure as MaxCut job response.

#### `POST /api/v1/jobs/portfolio`

Submit a Portfolio Optimization problem for solving.

**Request Body:**
```json
{
  "num_assets": 15,
  "risk_aversion": 0.5,
  "edge_profile": "ground",
  "strategy": "quality_optimized",
  "seed": 42
}
```

**Parameters:**
- `num_assets` (int, required): Number of assets (5-100)
- `risk_aversion` (float, optional): Risk aversion parameter (0.0-1.0), default: 0.5
- `edge_profile` (str, optional): Edge deployment profile
- `strategy` (str, optional): Routing strategy
- `seed` (int, optional): Random seed

**Response:** Same structure as MaxCut job response.

#### `POST /api/v1/jobs/comparative`

Run both classical and quantum solvers for side-by-side comparison.

**Request Body:**
```json
{
  "problem_type": "maxcut",
  "problem_size": 25,
  "edge_profile": "aerospace",
  "seed": 42
}
```

**Parameters:**
- `problem_type` (str, required): Problem type (maxcut/tsp/portfolio)
- `problem_size` (int, required): Problem size (5-100)
- `edge_profile` (str, optional): Edge deployment profile
- `seed` (int, optional): Random seed

**Response:**
```json
{
  "job_id": "650e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "problem_type": "maxcut",
  "problem_size": 25,
  "classical": {
    "solver": "classical",
    "solution": [1, 0, 1, ...],
    "cost": -123.45,
    "time_ms": 8.5,
    "energy_consumed_mj": 0.425,
    "is_valid": true,
    "solution_quality": 0.92
  },
  "quantum": {
    "solver": "quantum",
    "solution": [1, 0, 0, ...],
    "cost": -125.67,
    "time_ms": 15000.0,
    "energy_consumed_mj": 450.0,
    "is_valid": true,
    "solution_quality": 0.95
  },
  "speedup_factor": 0.0006,
  "energy_ratio": 1058.8,
  "quality_diff": 0.03,
  "recommendation": "quantum",
  "recommendation_reason": "Quantum solver provides 3% better solution quality",
  "timestamp": "2026-01-06T20:00:00"
}
```

---

### Routing Analysis Endpoints

#### `POST /api/v1/routing/analyze`

Analyze routing decision without executing the problem.

**Request Body:**
```json
{
  "problem_type": "maxcut",
  "problem_size": 20,
  "edge_profile": "aerospace",
  "strategy": "balanced"
}
```

**Response:**
```json
{
  "decision": "classical",
  "reasoning": "Problem size optimal for classical solver",
  "confidence": 0.95,
  "estimated_time_ms": 10.0,
  "estimated_energy_mj": 0.5,
  "alternative_solvers": ["quantum", "hybrid"],
  "constraints_satisfied": {
    "power_budget": true,
    "time_limit": true,
    "memory_limit": true
  }
}
```

#### `POST /api/v1/routing/explain`

Get detailed explanation of a routing decision (not yet implemented).

#### `POST /api/v1/routing/alternatives`

Get alternative solver options for a problem (not yet implemented).

---

## Request/Response Schemas

### Common Response Fields

All job submission endpoints return responses with these common fields:

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string (UUID) | Unique identifier for the job |
| `status` | string | Job status: "completed", "failed", "running" |
| `problem_type` | string | Type of problem: "maxcut", "tsp", "portfolio" |
| `problem_size` | integer | Size of the problem (nodes, cities, assets) |
| `solver_used` | string | Solver that executed: "classical", "quantum", "hybrid" |
| `routing_decision` | string | Routing decision made |
| `routing_reason` | string | Explanation of routing decision |
| `routing_confidence` | float | Confidence score (0.0-1.0) |
| `solution` | array | Binary or integer solution vector |
| `cost` | float | Objective function value |
| `time_ms` | float | Execution time in milliseconds |
| `energy_consumed_mj` | float | Energy consumption in millijoules |
| `is_valid` | boolean | Whether solution satisfies constraints |
| `solution_quality` | float | Quality metric (0.0-1.0) |
| `timestamp` | string (ISO 8601) | Submission timestamp |
| `error` | string (optional) | Error message if failed |

---

## Error Handling

The API uses standard HTTP status codes:

| Status Code | Meaning | Description |
|-------------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server-side error |

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 400,
  "timestamp": "2026-01-06T20:00:00"
}
```

**Common Error Scenarios:**

1. **Invalid Problem Size:**
```json
{
  "detail": "num_nodes must be between 5 and 100",
  "status_code": 422
}
```

2. **Invalid Edge Profile:**
```json
{
  "detail": "edge_profile must be one of ['aerospace', 'mobile', 'ground_server']",
  "status_code": 422
}
```

3. **Problem Generation Failed:**
```json
{
  "detail": "Failed to generate problem: Invalid parameters",
  "status_code": 500
}
```

---

## Code Examples

### Python with `requests`

```python
import requests

# Submit MaxCut problem
response = requests.post(
    "http://localhost:8000/api/v1/jobs/maxcut",
    json={
        "num_nodes": 30,
        "edge_probability": 0.3,
        "edge_profile": "aerospace",
        "strategy": "balanced",
        "seed": 42
    }
)

result = response.json()
print(f"Job ID: {result['job_id']}")
print(f"Solver: {result['solver_used']}")
print(f"Cost: {result['cost']}")
print(f"Time: {result['time_ms']} ms")
```

### Python with `httpx` (async)

```python
import httpx
import asyncio

async def submit_problem():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/jobs/maxcut",
            json={
                "num_nodes": 30,
                "edge_probability": 0.3,
                "edge_profile": "aerospace",
                "strategy": "balanced"
            }
        )
        return response.json()

result = asyncio.run(submit_problem())
print(result)
```

### cURL

```bash
# Submit MaxCut problem
curl -X POST "http://localhost:8000/api/v1/jobs/maxcut" \
  -H "Content-Type: application/json" \
  -d '{
    "num_nodes": 30,
    "edge_probability": 0.3,
    "edge_profile": "aerospace",
    "strategy": "balanced",
    "seed": 42
  }'

# Get system info
curl http://localhost:8000/api/v1/system/info

# Get edge profiles
curl http://localhost:8000/api/v1/config/edge-profiles

# Run comparative analysis
curl -X POST "http://localhost:8000/api/v1/jobs/comparative" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "maxcut",
    "problem_size": 10,
    "edge_profile": "aerospace",
    "seed": 42
  }'
```

### JavaScript (Node.js)

```javascript
const fetch = require('node-fetch');

async function submitMaxCut() {
  const response = await fetch('http://localhost:8000/api/v1/jobs/maxcut', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      num_nodes: 30,
      edge_probability: 0.3,
      edge_profile: 'aerospace',
      strategy: 'balanced',
      seed: 42
    })
  });
  
  const result = await response.json();
  console.log('Job ID:', result.job_id);
  console.log('Solver:', result.solver_used);
  console.log('Cost:', result.cost);
}

submitMaxCut();
```

---

## Testing

### Running Tests

The repository includes a comprehensive test suite for all API endpoints:

```bash
# Run all API tests
python test_api_endpoints.py
```

**Test Coverage:**
- ✅ Health check endpoint
- ✅ System information endpoint
- ✅ Edge profiles configuration
- ✅ Routing strategies configuration
- ✅ Routing analysis
- ✅ MaxCut job submission
- ✅ Comparative analysis

**Expected Output:**
```
================================================================================
QUANTUMEDGE PIPELINE API - ENDPOINT TESTS
================================================================================

Testing /health endpoint... ✓ PASSED
Testing /api/v1/system/info endpoint... ✓ PASSED
Testing /api/v1/config/edge-profiles endpoint... ✓ PASSED
Testing /api/v1/config/routing-strategies endpoint... ✓ PASSED
Testing /api/v1/routing/analyze endpoint... ✓ PASSED
Testing /api/v1/jobs/maxcut endpoint... ✓ PASSED
    Job ID: 550e8400-e29b-41d4-a716-446655440000
    Solver: classical
    Time: 2.00 ms
Testing /api/v1/jobs/comparative endpoint... ✓ PASSED
    Job ID: 650e8400-e29b-41d4-a716-446655440000
    Recommendation: quantum
    Speedup: 0.00x

================================================================================
RESULTS: 7 passed, 0 failed
================================================================================
```

### Interactive Testing

Use the built-in Swagger UI for interactive API testing:

1. Start the server:
   ```bash
   cd /home/user/webapp
   uvicorn src.api.main:app --reload
   ```

2. Open your browser to: `http://localhost:8000/docs`

3. Try out endpoints directly in the browser

---

## Additional Resources

- **GitHub Repository**: https://github.com/Gasta88/quantumedge-pipeline
- **Issue Tracker**: https://github.com/Gasta88/quantumedge-pipeline/issues
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## Changelog

### Version 0.1.0 (2026-01-06)
- ✅ Initial API implementation
- ✅ Job submission endpoints (MaxCut, TSP, Portfolio)
- ✅ Comparative analysis endpoint
- ✅ Routing analysis endpoints
- ✅ Configuration endpoints
- ✅ System monitoring endpoints
- ✅ Comprehensive test suite
- ✅ OpenAPI/Swagger documentation
