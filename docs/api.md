# 🔌 QuantumEdge Pipeline API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Schemas](#requestresponse-schemas)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Code Examples](#code-examples)
8. [OpenAPI Schema](#openapi-schema)

---

## Overview

The QuantumEdge Pipeline exposes a RESTful API built with **FastAPI** that allows developers to:
- Submit optimization problems for quantum-classical hybrid solving
- Query job status and retrieve results
- Run comparative analyses across multiple solvers
- Access historical performance metrics
- List supported problem types and solvers

**Base URL**: `http://localhost:8000` (development) or `https://api.quantumedge.io` (production)

**API Version**: `v1`

**Supported Formats**: JSON

**Interactive Documentation**: 
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Authentication

### Current Implementation (Development)

The development version currently operates **without authentication** for ease of testing and integration.

### Future Implementation (Production)

Production deployments will support multiple authentication methods:

#### 1. API Key Authentication (Recommended)

```http
GET /api/v1/jobs
Authorization: Bearer YOUR_API_KEY_HERE
```

**Obtain API Key:**
```bash
curl -X POST https://api.quantumedge.io/v1/auth/api-keys \
  -H "Content-Type: application/json" \
  -d '{
    "user_email": "your.email@example.com",
    "description": "Production API access"
  }'
```

#### 2. OAuth 2.0 (Enterprise)

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=YOUR_CLIENT_ID&
client_secret=YOUR_CLIENT_SECRET
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

#### 3. JWT Tokens (User Sessions)

```http
GET /api/v1/jobs
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## API Endpoints

### 1. Job Submission

#### `POST /api/v1/jobs/submit`

Submit an optimization problem for solving.

**Description**: Submit a problem to the QuantumEdge Pipeline. The system will automatically analyze the problem, select the optimal solver (classical, quantum, or hybrid), execute the computation, and return results.

**Request Body:**

```json
{
  "problem_type": "maxcut",
  "graph": {
    "nodes": 10,
    "edges": [
      [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
      [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]
    ],
    "weights": [1.0, 1.5, 2.0, 1.0, 1.8, 2.5, 1.2, 1.6, 1.9, 2.1]
  },
  "constraints": {
    "max_time_seconds": 60,
    "max_memory_mb": 2048,
    "prefer_quantum": false
  },
  "options": {
    "return_metrics": true,
    "enable_comparative": false,
    "quantum_shots": 1000
  }
}
```

**Request Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `problem_type` | string | ✅ | Type of optimization problem: `"maxcut"`, `"tsp"`, `"portfolio"`, `"graph_partition"` |
| `graph` | object | ⚠️ | Graph definition (required for graph problems) |
| `graph.nodes` | integer | ✅ | Number of nodes in the graph |
| `graph.edges` | array | ✅ | List of edges as `[source, target]` pairs |
| `graph.weights` | array | ❌ | Optional edge weights (defaults to 1.0) |
| `constraints` | object | ❌ | Resource and time constraints |
| `constraints.max_time_seconds` | integer | ❌ | Maximum allowed execution time (default: 300) |
| `constraints.max_memory_mb` | integer | ❌ | Maximum memory usage in MB (default: 4096) |
| `constraints.prefer_quantum` | boolean | ❌ | Prefer quantum solver when applicable (default: false) |
| `options` | object | ❌ | Additional execution options |
| `options.return_metrics` | boolean | ❌ | Include detailed metrics in response (default: true) |
| `options.enable_comparative` | boolean | ❌ | Run multiple solvers for comparison (default: false) |
| `options.quantum_shots` | integer | ❌ | Number of quantum measurement shots (default: 1000) |

**Response (200 OK):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "problem_type": "maxcut",
  "solver_type": "quantum_qaoa",
  "solution": {
    "cut_value": 12.5,
    "partition": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  },
  "metrics": {
    "execution_time_seconds": 8.742,
    "memory_usage_mb": 456.2,
    "energy_consumed_joules": 67.8,
    "solution_quality": 0.94,
    "quantum_circuit_depth": 12,
    "quantum_shots_used": 1000
  },
  "routing_decision": {
    "selected_solver": "quantum_qaoa",
    "confidence": 0.87,
    "reasoning": "Problem size (10 nodes) within quantum advantage zone. Graph is sparse (density: 0.22), favorable for QAOA."
  },
  "submitted_at": "2025-01-05T14:23:45.123Z",
  "completed_at": "2025-01-05T14:23:53.865Z"
}
```

**Response Codes:**

- `200 OK`: Job completed successfully
- `202 Accepted`: Job accepted and queued (for async processing)
- `400 Bad Request`: Invalid problem specification
- `422 Unprocessable Entity`: Valid JSON but invalid problem parameters
- `500 Internal Server Error`: Solver execution failed

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/jobs/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "maxcut",
    "graph": {
      "nodes": 10,
      "edges": [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,0]]
    },
    "constraints": {
      "max_time_seconds": 60,
      "max_memory_mb": 2048
    }
  }'
```

---

### 2. Job Status Query

#### `GET /api/v1/jobs/{job_id}`

Retrieve the status and results of a submitted job.

**Description**: Query the current status of a job by its unique ID. Returns job metadata, execution progress, and results (if completed).

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | UUID | ✅ | Unique identifier of the job (returned from submit endpoint) |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `include_circuit` | boolean | ❌ | Include quantum circuit details in response (default: false) |
| `include_raw_data` | boolean | ❌ | Include raw measurement data (default: false) |

**Response (200 OK - Job Completed):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "problem_type": "maxcut",
  "solver_type": "quantum_qaoa",
  "solution": {
    "cut_value": 12.5,
    "partition": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  },
  "metrics": {
    "execution_time_seconds": 8.742,
    "memory_usage_mb": 456.2,
    "energy_consumed_joules": 67.8,
    "solution_quality": 0.94
  },
  "submitted_at": "2025-01-05T14:23:45.123Z",
  "started_at": "2025-01-05T14:23:46.001Z",
  "completed_at": "2025-01-05T14:23:53.865Z"
}
```

**Response (200 OK - Job Running):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.65,
  "estimated_time_remaining_seconds": 15.2,
  "submitted_at": "2025-01-05T14:23:45.123Z",
  "started_at": "2025-01-05T14:23:46.001Z"
}
```

**Response (200 OK - Job Queued):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "queue_position": 3,
  "estimated_wait_time_seconds": 45,
  "submitted_at": "2025-01-05T14:23:45.123Z"
}
```

**Job Status Values:**

| Status | Description |
|--------|-------------|
| `queued` | Job accepted and waiting for execution |
| `running` | Job is currently being processed |
| `completed` | Job finished successfully |
| `failed` | Job execution failed (see `error` field) |
| `cancelled` | Job was cancelled by user |

**Response Codes:**

- `200 OK`: Job found and status returned
- `404 Not Found`: Job ID does not exist
- `500 Internal Server Error`: Error retrieving job status

**cURL Example:**

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000" \
  -H "Accept: application/json"
```

---

### 3. Comparative Analysis

#### `POST /api/v1/jobs/comparative`

Run the same problem on multiple solvers in parallel for comparison.

**Description**: Execute a problem using multiple solvers simultaneously (classical, quantum, hybrid) and return comparative performance metrics. Useful for benchmarking and selecting the best solver for your use case.

**Request Body:**

```json
{
  "problem_type": "tsp",
  "cities": [
    {"id": 0, "x": 0.0, "y": 0.0, "name": "A"},
    {"id": 1, "x": 1.0, "y": 2.0, "name": "B"},
    {"id": 2, "x": 3.0, "y": 1.0, "name": "C"},
    {"id": 3, "x": 4.0, "y": 3.0, "name": "D"},
    {"id": 4, "x": 2.0, "y": 4.0, "name": "E"}
  ],
  "solvers": [
    "classical_gurobi",
    "classical_ortools",
    "quantum_qaoa",
    "hybrid_vqe"
  ],
  "constraints": {
    "max_time_seconds": 120
  }
}
```

**Request Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `problem_type` | string | ✅ | Type of problem (same as submit endpoint) |
| `problem_data` | object | ✅ | Problem-specific data (varies by problem_type) |
| `solvers` | array | ✅ | List of solver IDs to compare |
| `constraints` | object | ❌ | Resource constraints applied to all solvers |

**Available Solvers:**

| Solver ID | Type | Description |
|-----------|------|-------------|
| `classical_gurobi` | Classical | Commercial MILP solver (exact) |
| `classical_ortools` | Classical | Google OR-Tools (heuristic) |
| `classical_networkx` | Classical | NetworkX graph algorithms |
| `quantum_qaoa` | Quantum | Quantum Approximate Optimization Algorithm |
| `quantum_vqe` | Quantum | Variational Quantum Eigensolver |
| `hybrid_vqe` | Hybrid | Quantum-assisted classical optimization |
| `rotonium_photonic` | Quantum | Rotonium photonic QPU (when available) |

**Response (200 OK):**

```json
{
  "comparison_id": "cmp_abc123def456",
  "problem_type": "tsp",
  "solvers_compared": 4,
  "results": [
    {
      "solver": "classical_gurobi",
      "status": "completed",
      "solution": {
        "tour": [0, 1, 3, 4, 2, 0],
        "total_distance": 9.23
      },
      "metrics": {
        "execution_time_seconds": 2.145,
        "memory_usage_mb": 234.5,
        "energy_consumed_joules": 145.2,
        "solution_quality": 1.00
      }
    },
    {
      "solver": "classical_ortools",
      "status": "completed",
      "solution": {
        "tour": [0, 2, 4, 3, 1, 0],
        "total_distance": 9.87
      },
      "metrics": {
        "execution_time_seconds": 1.532,
        "memory_usage_mb": 187.3,
        "energy_consumed_joules": 98.7,
        "solution_quality": 0.93
      }
    },
    {
      "solver": "quantum_qaoa",
      "status": "completed",
      "solution": {
        "tour": [0, 1, 3, 4, 2, 0],
        "total_distance": 9.23
      },
      "metrics": {
        "execution_time_seconds": 8.923,
        "memory_usage_mb": 512.1,
        "energy_consumed_joules": 72.4,
        "solution_quality": 1.00,
        "quantum_circuit_depth": 18,
        "quantum_shots": 1000
      }
    },
    {
      "solver": "hybrid_vqe",
      "status": "completed",
      "solution": {
        "tour": [0, 1, 3, 2, 4, 0],
        "total_distance": 9.45
      },
      "metrics": {
        "execution_time_seconds": 5.678,
        "memory_usage_mb": 389.4,
        "energy_consumed_joules": 84.3,
        "solution_quality": 0.98
      }
    }
  ],
  "summary": {
    "best_solver": "quantum_qaoa",
    "best_solution_quality": 1.00,
    "fastest_solver": "classical_ortools",
    "most_energy_efficient": "quantum_qaoa",
    "comparison_insights": [
      "Quantum QAOA matched classical optimal solution",
      "Quantum consumed 50% less energy than classical",
      "Classical Gurobi was fastest for this problem size"
    ]
  },
  "completed_at": "2025-01-05T14:28:15.456Z"
}
```

**Response Codes:**

- `200 OK`: Comparison completed successfully
- `400 Bad Request`: Invalid problem or solver specification
- `422 Unprocessable Entity`: Unsupported solver combination
- `500 Internal Server Error`: Comparison execution failed

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/jobs/comparative" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "tsp",
    "cities": [
      {"id": 0, "x": 0.0, "y": 0.0},
      {"id": 1, "x": 1.0, "y": 2.0},
      {"id": 2, "x": 3.0, "y": 1.0},
      {"id": 3, "x": 4.0, "y": 3.0},
      {"id": 4, "x": 2.0, "y": 4.0}
    ],
    "solvers": ["classical_gurobi", "quantum_qaoa", "hybrid_vqe"]
  }'
```

---

### 4. Query Performance Metrics

#### `GET /api/v1/metrics`

Query aggregated performance metrics across multiple jobs.

**Description**: Retrieve historical performance data for analysis, reporting, and optimization. Supports filtering by problem type, solver, date range, and other criteria.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `problem_type` | string | ❌ | Filter by problem type: `"maxcut"`, `"tsp"`, etc. |
| `solver_type` | string | ❌ | Filter by solver: `"quantum"`, `"classical"`, `"hybrid"` |
| `start_date` | string | ❌ | Start date (ISO 8601 format: `2025-01-01T00:00:00Z`) |
| `end_date` | string | ❌ | End date (ISO 8601 format) |
| `status` | string | ❌ | Filter by job status: `"completed"`, `"failed"`, etc. |
| `limit` | integer | ❌ | Maximum number of results (default: 100, max: 1000) |
| `offset` | integer | ❌ | Pagination offset (default: 0) |
| `aggregation` | string | ❌ | Aggregation level: `"daily"`, `"weekly"`, `"monthly"`, `"none"` |

**Response (200 OK):**

```json
{
  "query": {
    "problem_type": "maxcut",
    "solver_type": "quantum",
    "start_date": "2025-01-01T00:00:00Z",
    "end_date": "2025-01-31T23:59:59Z"
  },
  "total_jobs": 342,
  "summary": {
    "avg_execution_time_seconds": 7.832,
    "avg_solution_quality": 0.927,
    "avg_energy_consumed_joules": 68.4,
    "success_rate": 0.964,
    "total_energy_saved_vs_classical_joules": 15420.5
  },
  "aggregated_data": [
    {
      "date": "2025-01-01",
      "jobs_count": 15,
      "avg_execution_time": 8.123,
      "avg_quality": 0.915
    },
    {
      "date": "2025-01-02",
      "jobs_count": 18,
      "avg_execution_time": 7.654,
      "avg_quality": 0.932
    }
    // ... more daily aggregates
  ],
  "solver_comparison": {
    "quantum_qaoa": {
      "jobs": 198,
      "avg_execution_time": 8.234,
      "avg_quality": 0.934,
      "success_rate": 0.970
    },
    "quantum_vqe": {
      "jobs": 144,
      "avg_execution_time": 7.321,
      "avg_quality": 0.918,
      "success_rate": 0.956
    }
  },
  "pagination": {
    "total": 342,
    "limit": 100,
    "offset": 0,
    "has_more": true
  }
}
```

**Response Codes:**

- `200 OK`: Metrics retrieved successfully
- `400 Bad Request`: Invalid query parameters
- `500 Internal Server Error`: Database query failed

**cURL Example:**

```bash
curl -X GET "http://localhost:8000/api/v1/metrics?problem_type=maxcut&solver_type=quantum&start_date=2025-01-01&end_date=2025-01-31" \
  -H "Accept: application/json"
```

---

### 5. List Supported Problem Types

#### `GET /api/v1/problems`

List all supported problem types and their specifications.

**Description**: Retrieve information about available optimization problem types, including required parameters, constraints, and solver compatibility.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | ❌ | Filter by category: `"graph"`, `"combinatorial"`, `"continuous"` |

**Response (200 OK):**

```json
{
  "problems": [
    {
      "problem_type": "maxcut",
      "name": "Maximum Cut",
      "category": "graph",
      "description": "Partition graph nodes to maximize cut edges",
      "complexity": "NP-Complete",
      "quantum_advantage_zone": {
        "min_nodes": 8,
        "max_nodes": 50,
        "graph_density_range": [0.1, 0.5]
      },
      "required_parameters": [
        "graph.nodes",
        "graph.edges"
      ],
      "optional_parameters": [
        "graph.weights"
      ],
      "supported_solvers": [
        "classical_gurobi",
        "classical_networkx",
        "quantum_qaoa",
        "quantum_vqe",
        "hybrid_vqe"
      ],
      "example_request": {
        "problem_type": "maxcut",
        "graph": {
          "nodes": 10,
          "edges": [[0, 1], [1, 2]]
        }
      }
    },
    {
      "problem_type": "tsp",
      "name": "Traveling Salesman Problem",
      "category": "combinatorial",
      "description": "Find shortest tour visiting all cities",
      "complexity": "NP-Hard",
      "quantum_advantage_zone": {
        "min_cities": 5,
        "max_cities": 30
      },
      "required_parameters": [
        "cities"
      ],
      "optional_parameters": [
        "distance_matrix",
        "start_city"
      ],
      "supported_solvers": [
        "classical_gurobi",
        "classical_ortools",
        "quantum_qaoa",
        "hybrid_vqe"
      ],
      "example_request": {
        "problem_type": "tsp",
        "cities": [
          {"id": 0, "x": 0.0, "y": 0.0},
          {"id": 1, "x": 1.0, "y": 2.0}
        ]
      }
    },
    {
      "problem_type": "portfolio",
      "name": "Portfolio Optimization",
      "category": "continuous",
      "description": "Optimize asset allocation under risk constraints",
      "complexity": "Convex (classical), QUBO (quantum)",
      "quantum_advantage_zone": {
        "min_assets": 10,
        "max_assets": 100
      },
      "required_parameters": [
        "assets",
        "expected_returns",
        "covariance_matrix"
      ],
      "optional_parameters": [
        "risk_tolerance",
        "sector_constraints",
        "budget"
      ],
      "supported_solvers": [
        "classical_scipy",
        "classical_gurobi",
        "quantum_vqe",
        "hybrid_vqe"
      ],
      "example_request": {
        "problem_type": "portfolio",
        "assets": ["AAPL", "GOOGL", "MSFT"],
        "expected_returns": [0.12, 0.15, 0.10],
        "covariance_matrix": [[0.04, 0.01, 0.02], [0.01, 0.06, 0.01], [0.02, 0.01, 0.03]]
      }
    },
    {
      "problem_type": "graph_partition",
      "name": "Graph Partitioning",
      "category": "graph",
      "description": "Partition graph into k balanced subgraphs minimizing cut edges",
      "complexity": "NP-Complete",
      "quantum_advantage_zone": {
        "min_nodes": 10,
        "max_nodes": 100
      },
      "required_parameters": [
        "graph.nodes",
        "graph.edges",
        "num_partitions"
      ],
      "optional_parameters": [
        "balance_tolerance"
      ],
      "supported_solvers": [
        "classical_networkx",
        "classical_metis",
        "quantum_qaoa",
        "hybrid_vqe"
      ],
      "example_request": {
        "problem_type": "graph_partition",
        "graph": {
          "nodes": 20,
          "edges": [[0, 1], [1, 2]]
        },
        "num_partitions": 4
      }
    }
  ],
  "total_problem_types": 4
}
```

**Response Codes:**

- `200 OK`: Problem types retrieved successfully
- `500 Internal Server Error`: Error retrieving problem types

**cURL Example:**

```bash
curl -X GET "http://localhost:8000/api/v1/problems" \
  -H "Accept: application/json"
```

---

### 6. Health Check

#### `GET /api/v1/health`

Check API and system health status.

**Description**: Returns health status of the API, database, quantum backends, and other system components.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-05T14:35:22.123Z",
  "components": {
    "api": {
      "status": "up",
      "latency_ms": 1.23
    },
    "database": {
      "status": "up",
      "connection_pool": {
        "active": 3,
        "idle": 7,
        "max": 10
      }
    },
    "quantum_backends": {
      "qiskit_aer": {
        "status": "available",
        "max_qubits": 20
      },
      "rotonium_photonic": {
        "status": "available",
        "queue_length": 2,
        "avg_wait_time_seconds": 15
      }
    },
    "cache": {
      "status": "up",
      "hit_rate": 0.87
    }
  },
  "system_metrics": {
    "cpu_usage_percent": 34.5,
    "memory_usage_mb": 1245.8,
    "disk_usage_percent": 42.3
  }
}
```

**Response Codes:**

- `200 OK`: System healthy
- `503 Service Unavailable`: System degraded or unavailable

**cURL Example:**

```bash
curl -X GET "http://localhost:8000/api/v1/health" \
  -H "Accept: application/json"
```

---

## Request/Response Schemas

### Common Data Types

#### Problem Graph

```typescript
{
  nodes: number;              // Number of nodes
  edges: [number, number][];  // Edge list (source, target)
  weights?: number[];         // Optional edge weights (default: 1.0)
  directed?: boolean;         // Directed graph (default: false)
}
```

#### Optimization Constraints

```typescript
{
  max_time_seconds?: number;     // Max execution time (default: 300)
  max_memory_mb?: number;        // Max memory usage (default: 4096)
  max_energy_joules?: number;    // Max energy consumption
  prefer_quantum?: boolean;      // Prefer quantum solver (default: false)
  quality_threshold?: number;    // Min solution quality 0-1 (default: 0.8)
}
```

#### Job Metrics

```typescript
{
  execution_time_seconds: number;
  memory_usage_mb: number;
  energy_consumed_joules: number;
  solution_quality: number;         // 0-1 scale (1 = optimal)
  quantum_circuit_depth?: number;   // If quantum solver used
  quantum_shots_used?: number;      // If quantum solver used
  classical_iterations?: number;    // If classical solver used
}
```

#### Solution Object (varies by problem type)

**MaxCut Solution:**
```typescript
{
  cut_value: number;           // Total weight of cut edges
  partition: number[];         // Node assignments (0 or 1)
  cut_edges: [number, number][]; // List of edges in the cut
}
```

**TSP Solution:**
```typescript
{
  tour: number[];              // Ordered list of city IDs
  total_distance: number;      // Total tour distance
  visit_order: string[];       // City names in order (if provided)
}
```

**Portfolio Solution:**
```typescript
{
  allocation: Record<string, number>;  // Asset ID -> weight
  expected_return: number;             // Expected annual return
  portfolio_risk: number;              // Portfolio variance
  sharpe_ratio: number;                // Risk-adjusted return
}
```

---

## Error Handling

### Standard Error Response

All error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_PROBLEM_TYPE",
    "message": "Problem type 'maxcuttt' is not supported. Did you mean 'maxcut'?",
    "details": {
      "supported_types": ["maxcut", "tsp", "portfolio", "graph_partition"],
      "provided_type": "maxcuttt"
    },
    "timestamp": "2025-01-05T14:40:12.456Z",
    "request_id": "req_abc123def456"
  }
}
```

### Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_REQUEST` | Malformed request body or parameters |
| 400 | `INVALID_PROBLEM_TYPE` | Unsupported problem type |
| 400 | `INVALID_GRAPH` | Invalid graph structure |
| 400 | `MISSING_REQUIRED_FIELD` | Required field not provided |
| 401 | `UNAUTHORIZED` | Missing or invalid authentication |
| 403 | `FORBIDDEN` | Insufficient permissions |
| 404 | `JOB_NOT_FOUND` | Job ID does not exist |
| 422 | `UNPROCESSABLE_ENTITY` | Valid JSON but invalid parameters |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_SERVER_ERROR` | Unexpected server error |
| 503 | `SERVICE_UNAVAILABLE` | System overloaded or under maintenance |

---

## Rate Limiting

### Current Implementation (Development)

No rate limiting in development mode.

### Future Implementation (Production)

**Rate Limits (per API key):**

| Tier | Requests/Minute | Requests/Day | Concurrent Jobs |
|------|-----------------|--------------|-----------------|
| **Free** | 60 | 1,000 | 2 |
| **Developer** | 300 | 10,000 | 5 |
| **Pro** | 1,000 | 50,000 | 20 |
| **Enterprise** | Custom | Custom | Custom |

**Rate Limit Headers:**

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 285
X-RateLimit-Reset: 1704467400
```

**Rate Limit Exceeded Response (429):**

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please wait 45 seconds before retrying.",
    "retry_after_seconds": 45,
    "limit": 300,
    "reset_at": "2025-01-05T15:00:00Z"
  }
}
```

---

## Code Examples

### Python

```python
import httpx
import asyncio

async def solve_maxcut():
    """Example: Submit MaxCut problem and retrieve results."""
    
    async with httpx.AsyncClient() as client:
        # Submit problem
        response = await client.post(
            "http://localhost:8000/api/v1/jobs/submit",
            json={
                "problem_type": "maxcut",
                "graph": {
                    "nodes": 10,
                    "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                              [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]]
                },
                "constraints": {
                    "max_time_seconds": 60
                }
            },
            timeout=120.0
        )
        response.raise_for_status()
        result = response.json()
        
        job_id = result["job_id"]
        print(f"Job submitted: {job_id}")
        print(f"Solver: {result['solver_type']}")
        print(f"Cut value: {result['solution']['cut_value']}")
        print(f"Execution time: {result['metrics']['execution_time_seconds']:.2f}s")
        
        return result

# Run
asyncio.run(solve_maxcut())
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function solveTSP() {
  try {
    // Submit TSP problem
    const response = await axios.post(
      'http://localhost:8000/api/v1/jobs/submit',
      {
        problem_type: 'tsp',
        cities: [
          { id: 0, x: 0.0, y: 0.0 },
          { id: 1, x: 1.0, y: 2.0 },
          { id: 2, x: 3.0, y: 1.0 },
          { id: 3, x: 4.0, y: 3.0 },
          { id: 4, x: 2.0, y: 4.0 }
        ],
        constraints: {
          max_time_seconds: 60
        }
      }
    );
    
    const result = response.data;
    console.log(`Job ID: ${result.job_id}`);
    console.log(`Tour: ${result.solution.tour}`);
    console.log(`Distance: ${result.solution.total_distance}`);
    console.log(`Time: ${result.metrics.execution_time_seconds}s`);
    
    return result;
  } catch (error) {
    console.error('Error:', error.response.data);
    throw error;
  }
}

solveTSP();
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

type JobRequest struct {
    ProblemType string                 `json:"problem_type"`
    Graph       map[string]interface{} `json:"graph"`
    Constraints map[string]int         `json:"constraints"`
}

type JobResponse struct {
    JobID      string                 `json:"job_id"`
    Status     string                 `json:"status"`
    SolverType string                 `json:"solver_type"`
    Solution   map[string]interface{} `json:"solution"`
}

func main() {
    // Create request
    reqBody := JobRequest{
        ProblemType: "maxcut",
        Graph: map[string]interface{}{
            "nodes": 10,
            "edges": [][]int{{0, 1}, {1, 2}, {2, 3}},
        },
        Constraints: map[string]int{
            "max_time_seconds": 60,
        },
    }
    
    jsonData, _ := json.Marshal(reqBody)
    
    // Submit job
    resp, err := http.Post(
        "http://localhost:8000/api/v1/jobs/submit",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()
    
    // Parse response
    body, _ := ioutil.ReadAll(resp.Body)
    var result JobResponse
    json.Unmarshal(body, &result)
    
    fmt.Printf("Job ID: %s\n", result.JobID)
    fmt.Printf("Solver: %s\n", result.SolverType)
    fmt.Printf("Status: %s\n", result.Status)
}
```

---

## OpenAPI Schema

The complete OpenAPI 3.0 schema is available at:

**Swagger UI**: http://localhost:8000/docs

**OpenAPI JSON**: http://localhost:8000/openapi.json

### Sample OpenAPI Schema (Excerpt)

```yaml
openapi: 3.0.0
info:
  title: QuantumEdge Pipeline API
  version: 1.0.0
  description: Quantum-classical hybrid optimization API
  contact:
    email: api-support@quantumedge.io
  license:
    name: MIT

servers:
  - url: http://localhost:8000/api/v1
    description: Development server
  - url: https://api.quantumedge.io/api/v1
    description: Production server

paths:
  /jobs/submit:
    post:
      summary: Submit optimization problem
      operationId: submitJob
      tags:
        - Jobs
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/JobSubmitRequest'
      responses:
        '200':
          description: Job completed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobResponse'
        '400':
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

components:
  schemas:
    JobSubmitRequest:
      type: object
      required:
        - problem_type
      properties:
        problem_type:
          type: string
          enum: [maxcut, tsp, portfolio, graph_partition]
        graph:
          $ref: '#/components/schemas/Graph'
        constraints:
          $ref: '#/components/schemas/Constraints'
    
    Graph:
      type: object
      required:
        - nodes
        - edges
      properties:
        nodes:
          type: integer
          minimum: 2
        edges:
          type: array
          items:
            type: array
            items:
              type: integer
            minItems: 2
            maxItems: 2
    
    JobResponse:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [queued, running, completed, failed]
        solver_type:
          type: string
        solution:
          type: object
        metrics:
          $ref: '#/components/schemas/Metrics'
```

---

## Summary

The QuantumEdge Pipeline API provides a **production-ready interface** for quantum-classical hybrid optimization:

✅ **Simple REST API**: Standard HTTP/JSON interface  
✅ **Comprehensive Documentation**: OpenAPI/Swagger specs  
✅ **Multiple Solvers**: Classical, quantum, and hybrid backends  
✅ **Performance Metrics**: Detailed execution and energy tracking  
✅ **Comparative Analysis**: Side-by-side solver benchmarking  
✅ **Future-Ready**: Authentication, rate limiting, and enterprise features planned  

For technical support or API questions:
- **Documentation**: https://docs.quantumedge.io
- **API Status**: https://status.quantumedge.io
- **Support Email**: api-support@quantumedge.io

---

*Last updated: 2025-01-05*  
*API Version: 1.0.0*  
*Maintained by: QuantumEdge Team*
