"""
FastAPI REST API for QuantumEdge Pipeline.

This module provides a comprehensive REST API for submitting optimization problems,
tracking job execution, and analyzing routing decisions in the QuantumEdge Pipeline.

The API exposes:
- Job submission endpoints (MaxCut, TSP, Portfolio)
- Job status and result retrieval
- Comparative analysis (classical vs quantum)
- Routing analysis and explanation
- System health and configuration

All endpoints are documented via OpenAPI/Swagger at /docs.

Example Usage:
    # Start the server
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    
    # Submit a MaxCut problem
    curl -X POST "http://localhost:8000/api/v1/jobs/maxcut" \
      -H "Content-Type: application/json" \
      -d '{"num_nodes": 30, "edge_probability": 0.3}'
    
    # Check system health
    curl http://localhost:8000/health
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from uuid import uuid4
import logging
import traceback
import time
import numpy as np

# Import application components
from src.config import settings
from src.api.orchestrator import JobOrchestrator
from src.router.quantum_router import QuantumRouter, RoutingStrategy, RoutingPreferences
from src.router.edge_simulator import EdgeEnvironment, DeploymentProfile
from src.problems.maxcut import MaxCutProblem
from src.problems.tsp import TSPProblem
from src.problems.portfolio import PortfolioProblem

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.api.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class MaxCutJobRequest(BaseModel):
    """Request model for MaxCut problem submission."""
    num_nodes: int = Field(
        ..., 
        ge=5, 
        le=100, 
        description="Number of nodes in the graph",
        example=30
    )
    edge_probability: float = Field(
        0.3, 
        ge=0.0, 
        le=1.0, 
        description="Probability of edge existence between nodes",
        example=0.3
    )
    edge_profile: str = Field(
        "aerospace", 
        description="Edge deployment profile: aerospace, mobile, ground_server",
        example="aerospace"
    )
    strategy: str = Field(
        "balanced", 
        description="Routing strategy: balanced, energy_optimized, latency_optimized, quality_optimized",
        example="balanced"
    )
    seed: Optional[int] = Field(
        None, 
        description="Random seed for reproducibility",
        example=42
    )

    @validator('edge_profile')
    def validate_edge_profile(cls, v):
        valid_profiles = ['aerospace', 'mobile', 'ground_server']
        if v not in valid_profiles:
            raise ValueError(f"edge_profile must be one of {valid_profiles}")
        return v

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['balanced', 'energy_optimized', 'latency_optimized', 'quality_optimized']
        if v not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        return v


class TSPJobRequest(BaseModel):
    """Request model for TSP problem submission."""
    num_cities: int = Field(
        ..., 
        ge=5, 
        le=50, 
        description="Number of cities in the tour",
        example=15
    )
    euclidean: bool = Field(
        True, 
        description="Use Euclidean distances (2D plane)",
        example=True
    )
    edge_profile: str = Field(
        "aerospace", 
        description="Edge deployment profile",
        example="aerospace"
    )
    strategy: str = Field(
        "balanced", 
        description="Routing strategy",
        example="balanced"
    )
    seed: Optional[int] = Field(
        None, 
        description="Random seed for reproducibility",
        example=42
    )

    @validator('edge_profile')
    def validate_edge_profile(cls, v):
        valid_profiles = ['aerospace', 'mobile', 'ground_server']
        if v not in valid_profiles:
            raise ValueError(f"edge_profile must be one of {valid_profiles}")
        return v

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['balanced', 'energy_optimized', 'latency_optimized', 'quality_optimized']
        if v not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        return v


class PortfolioJobRequest(BaseModel):
    """Request model for Portfolio optimization problem submission."""
    num_assets: int = Field(
        ..., 
        ge=5, 
        le=100, 
        description="Number of assets in portfolio",
        example=20
    )
    risk_aversion: float = Field(
        0.5, 
        ge=0.0, 
        le=1.0, 
        description="Risk aversion parameter (0=risk-seeking, 1=risk-averse)",
        example=0.5
    )
    edge_profile: str = Field(
        "ground_server", 
        description="Edge deployment profile",
        example="ground_server"
    )
    strategy: str = Field(
        "balanced", 
        description="Routing strategy",
        example="balanced"
    )
    seed: Optional[int] = Field(
        None, 
        description="Random seed for reproducibility",
        example=42
    )

    @validator('edge_profile')
    def validate_edge_profile(cls, v):
        valid_profiles = ['aerospace', 'mobile', 'ground_server']
        if v not in valid_profiles:
            raise ValueError(f"edge_profile must be one of {valid_profiles}")
        return v

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['balanced', 'energy_optimized', 'latency_optimized', 'quality_optimized']
        if v not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        return v


class JobResponse(BaseModel):
    """Response model for job execution results."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: completed, failed")
    problem_type: str = Field(..., description="Type of problem: maxcut, tsp, portfolio")
    problem_size: int = Field(..., description="Problem size (nodes/cities/assets)")
    solver_used: Optional[str] = Field(None, description="Solver used: classical, quantum, hybrid")
    routing_decision: Optional[str] = Field(None, description="Routing decision made")
    routing_reason: Optional[str] = Field(None, description="Reasoning for routing decision")
    routing_confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    solution: Optional[Any] = Field(None, description="Solution found")
    cost: Optional[float] = Field(None, description="Objective function value")
    time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    energy_consumed_mj: Optional[float] = Field(None, description="Energy consumed in millijoules")
    is_valid: Optional[bool] = Field(None, description="Whether solution is valid")
    solution_quality: Optional[float] = Field(None, description="Solution quality score (0-1)")
    timestamp: datetime = Field(..., description="Job completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "problem_type": "maxcut",
                "problem_size": 30,
                "solver_used": "quantum",
                "routing_decision": "quantum",
                "routing_confidence": 0.75,
                "cost": -15.42,
                "time_ms": 2145.5,
                "energy_consumed_mj": 120.5,
                "is_valid": True,
                "solution_quality": 0.92,
                "timestamp": "2025-01-06T20:30:00"
            }
        }


class ComparativeJobRequest(BaseModel):
    """Request model for comparative classical vs quantum analysis."""
    problem_type: str = Field(..., description="Problem type: maxcut, tsp, portfolio")
    problem_size: int = Field(..., ge=5, le=100, description="Problem size")
    edge_profile: str = Field("aerospace", description="Edge deployment profile")
    seed: Optional[int] = Field(None, description="Random seed")

    @validator('problem_type')
    def validate_problem_type(cls, v):
        valid_types = ['maxcut', 'tsp', 'portfolio']
        if v not in valid_types:
            raise ValueError(f"problem_type must be one of {valid_types}")
        return v

    @validator('edge_profile')
    def validate_edge_profile(cls, v):
        valid_profiles = ['aerospace', 'mobile', 'ground_server']
        if v not in valid_profiles:
            raise ValueError(f"edge_profile must be one of {valid_profiles}")
        return v


class ComparativeJobResponse(BaseModel):
    """Response model for comparative analysis results."""
    job_id: str
    success: bool
    problem_type: str
    problem_size: int
    classical: Dict[str, Any]
    quantum: Dict[str, Any]
    speedup_factor: Optional[float] = None
    energy_ratio: Optional[float] = None
    quality_diff: Optional[float] = None
    recommendation: str
    recommendation_reason: str
    timestamp: datetime


class RoutingAnalysisRequest(BaseModel):
    """Request model for routing analysis without execution."""
    problem_type: str = Field(..., description="Problem type: maxcut, tsp, portfolio")
    problem_size: int = Field(..., ge=5, le=100, description="Problem size")
    edge_profile: str = Field("aerospace", description="Deployment profile")
    strategy: str = Field("balanced", description="Routing strategy")

    @validator('problem_type')
    def validate_problem_type(cls, v):
        valid_types = ['maxcut', 'tsp', 'portfolio']
        if v not in valid_types:
            raise ValueError(f"problem_type must be one of {valid_types}")
        return v

    @validator('edge_profile')
    def validate_edge_profile(cls, v):
        valid_profiles = ['aerospace', 'mobile', 'ground_server']
        if v not in valid_profiles:
            raise ValueError(f"edge_profile must be one of {valid_profiles}")
        return v

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['balanced', 'energy_optimized', 'latency_optimized', 'quality_optimized']
        if v not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        return v


class RoutingAnalysisResponse(BaseModel):
    """Response model for routing analysis."""
    decision: str
    reasoning: str
    confidence: float
    estimated_time_ms: int
    estimated_energy_mj: float
    alternative_options: List[Dict[str, Any]]
    strategy_used: str
    problem_analysis: Dict[str, Any]
    resource_constraints: Dict[str, Any]
    performance_predictions: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    environment: str
    timestamp: str
    components: Dict[str, str]


class SystemInfoResponse(BaseModel):
    """Response model for system information."""
    application: Dict[str, Any]
    configuration: Dict[str, Any]
    capabilities: Dict[str, Any]


class EdgeProfileResponse(BaseModel):
    """Response model for edge profile information."""
    profile_name: str
    power_budget_watts: float
    thermal_limit_celsius: float
    memory_mb: int
    cpu_cores: int
    max_execution_time_sec: int
    network_latency_ms: int


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    description="""
    **QuantumEdge Pipeline REST API**
    
    A production-ready quantum-classical hybrid optimization framework designed for 
    edge computing environments. This API enables:
    
    - 🚀 **Problem Submission**: Submit MaxCut, TSP, and Portfolio optimization problems
    - 🧠 **Intelligent Routing**: Automatic solver selection (classical vs quantum)
    - 📊 **Comparative Analysis**: Side-by-side performance evaluation
    - 🔍 **Routing Analysis**: Understand routing decisions without execution
    - 💡 **Edge Computing**: Optimized for aerospace, mobile, and ground server deployments
    
    ## Key Features
    
    - **Rotonium Integration**: Optimized for room-temperature photonic quantum processors
    - **Resource Awareness**: Respects power, memory, and thermal constraints
    - **Multiple Strategies**: Balanced, energy-optimized, latency-optimized, quality-optimized
    - **Transparent Decisions**: Detailed reasoning and confidence scores
    
    ## Quick Start
    
    1. Submit a MaxCut problem: `POST /api/v1/jobs/maxcut`
    2. Check job status: `GET /api/v1/jobs/{job_id}`
    3. Analyze routing: `POST /api/v1/routing/analyze`
    
    ## Support
    
    For documentation, visit: https://github.com/Gasta88/quantumedge-pipeline
    """,
    debug=settings.api.debug,
    docs_url=settings.api.docs_url,
    redoc_url=settings.api.redoc_url,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=settings.api.cors_allow_credentials,
    allow_methods=settings.api.cors_allow_methods,
    allow_headers=settings.api.cors_allow_headers,
)


# =============================================================================
# Middleware & Error Handlers
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing information."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Response: {response.status_code} - {process_time:.2f}ms")
    
    # Add custom headers
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed error messages."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.api.debug else "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )


# =============================================================================
# Health Check & System Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health and service availability.
    
    Returns the current status of all system components including:
    - API service status
    - Orchestrator availability
    - Router availability
    - Database connection (if enabled)
    
    This endpoint is useful for:
    - Load balancer health checks
    - Monitoring and alerting
    - Deployment verification
    """
    try:
        # Database status (disabled due to async driver requirements)
        db_status = "disabled"
        
        return HealthResponse(
            status="healthy",
            version=settings.api.version,
            environment=settings.environment,
            timestamp=datetime.now().isoformat(),
            components={
                "api": "ready",
                "orchestrator": "ready",
                "router": "ready",
                "database": db_status
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/api/v1/system/info", response_model=SystemInfoResponse, tags=["System"])
async def get_system_info():
    """
    Get comprehensive system information and configuration.
    
    Returns details about:
    - Application version and environment
    - Configuration settings
    - Available capabilities and features
    - Supported problem types
    - Edge profiles and routing strategies
    """
    return SystemInfoResponse(
        application={
            "name": settings.api.title,
            "version": settings.api.version,
            "environment": settings.environment,
            "debug": settings.api.debug
        },
        configuration={
            "quantum_backend": settings.quantum.backend,
            "quantum_shots": settings.quantum.shots,
            "quantum_max_qubits": settings.quantum.max_qubits,
            "database_enabled": bool(settings.database),
            "default_edge_profile": settings.edge.default_profile
        },
        capabilities={
            "problem_types": ["maxcut", "tsp", "portfolio"],
            "solver_types": ["classical", "quantum", "hybrid"],
            "edge_profiles": ["aerospace", "mobile", "ground_server"],
            "routing_strategies": [
                "balanced", 
                "energy_optimized", 
                "latency_optimized", 
                "quality_optimized"
            ],
            "max_problem_sizes": {
                "maxcut": 100,
                "tsp": 50,
                "portfolio": 100
            }
        }
    )


@app.get("/api/v1/system/stats", tags=["System"])
async def get_system_stats():
    """
    Get system execution statistics.
    
    Returns aggregated statistics about:
    - Total jobs executed
    - Success/failure rates
    - Average execution times
    - Solver usage distribution
    """
    try:
        # Create a temporary orchestrator to get stats
        orchestrator = JobOrchestrator(enable_db=False)
        stats = orchestrator.get_statistics()
        
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Job Submission Endpoints
# =============================================================================

@app.post("/api/v1/jobs/maxcut", response_model=JobResponse, tags=["Jobs"])
async def submit_maxcut_job(request: MaxCutJobRequest):
    """
    Submit a MaxCut optimization problem for solving.
    
    The MaxCut problem seeks to partition graph nodes into two sets such that
    the number (or weight) of edges between the sets is maximized. This is an
    NP-hard problem with applications in:
    - Network design and clustering
    - Circuit layout optimization
    - Image segmentation
    - Community detection in social networks
    
    **Process Flow**:
    1. Generate random MaxCut instance with specified parameters
    2. Analyze problem characteristics (size, structure, complexity)
    3. Route to optimal solver (classical/quantum/hybrid) based on:
       - Problem size and structure
       - Edge profile resource constraints
       - Routing strategy preferences
    4. Execute solver and track performance metrics
    5. Validate solution and calculate quality score
    6. Return comprehensive results with routing explanation
    
    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/jobs/maxcut" \\
      -H "Content-Type: application/json" \\
      -d '{
        "num_nodes": 30,
        "edge_probability": 0.3,
        "edge_profile": "aerospace",
        "strategy": "balanced",
        "seed": 42
      }'
    ```
    """
    try:
        logger.info(f"Received MaxCut job request: {request.dict()}")
        
        # Generate problem
        problem = MaxCutProblem(num_nodes=request.num_nodes)
        problem.generate(edge_probability=request.edge_probability, seed=request.seed)
        
        logger.info(f"Generated MaxCut problem: {request.num_nodes} nodes, "
                   f"edge_probability={request.edge_probability}")
        
        # Create orchestrator (database disabled for now - async driver issue)
        orchestrator = JobOrchestrator(
            enable_db=False,  # Disabled: async driver required
            db_url=None,
            strategy=request.strategy
        )
        
        # Execute job
        result = orchestrator.execute_job(
            problem=problem,
            edge_profile=request.edge_profile,
            strategy=request.strategy
        )
        
        logger.info(f"MaxCut job completed: {result['job_id']}, "
                   f"solver={result.get('solver_used')}, "
                   f"time={result.get('time_ms')}ms")
        
        # Convert to response model
        return JobResponse(
            job_id=result['job_id'],
            status="completed" if result['success'] else "failed",
            problem_type=result['problem_type'],
            problem_size=result['problem_size'],
            solver_used=result.get('solver_used'),
            routing_decision=result.get('routing_decision'),
            routing_reason=result.get('routing_reason'),
            routing_confidence=result.get('routing_confidence'),
            solution=result.get('solution'),
            cost=result.get('cost'),
            time_ms=result.get('time_ms'),
            energy_consumed_mj=result.get('energy_consumed_mj'),
            is_valid=result.get('is_valid'),
            solution_quality=result.get('solution_quality'),
            timestamp=result['timestamp'],
            error=result.get('error')
        )
        
    except ValueError as e:
        logger.error(f"Invalid MaxCut job request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"MaxCut job execution failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/jobs/tsp", response_model=JobResponse, tags=["Jobs"])
async def submit_tsp_job(request: TSPJobRequest):
    """
    Submit a Traveling Salesman Problem (TSP) for solving.
    
    The TSP seeks to find the shortest route visiting all cities exactly once
    and returning to the start. This is a classic NP-hard problem with applications in:
    - Logistics and route planning
    - Manufacturing (drill routing, PCB assembly)
    - DNA sequencing
    - Telescope observation scheduling
    
    **Process Flow**:
    1. Generate random TSP instance with specified number of cities
    2. Create distance matrix (Euclidean or general metric)
    3. Analyze problem and route to optimal solver
    4. Execute and return optimal/near-optimal tour
    5. Calculate tour length and quality metrics
    
    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/jobs/tsp" \\
      -H "Content-Type: application/json" \\
      -d '{
        "num_cities": 15,
        "euclidean": true,
        "edge_profile": "aerospace",
        "strategy": "latency_optimized"
      }'
    ```
    """
    try:
        logger.info(f"Received TSP job request: {request.dict()}")
        
        # Generate problem
        problem = TSPProblem(num_cities=request.num_cities)
        problem.generate(euclidean=request.euclidean, seed=request.seed)
        
        logger.info(f"Generated TSP problem: {request.num_cities} cities, "
                   f"euclidean={request.euclidean}")
        
        # Create orchestrator (database disabled for now - async driver issue)
        orchestrator = JobOrchestrator(
            enable_db=False,  # Disabled: async driver required
            db_url=None,
            strategy=request.strategy
        )
        
        # Execute job
        result = orchestrator.execute_job(
            problem=problem,
            edge_profile=request.edge_profile,
            strategy=request.strategy
        )
        
        logger.info(f"TSP job completed: {result['job_id']}, "
                   f"solver={result.get('solver_used')}, "
                   f"time={result.get('time_ms')}ms")
        
        # Convert to response model
        return JobResponse(
            job_id=result['job_id'],
            status="completed" if result['success'] else "failed",
            problem_type=result['problem_type'],
            problem_size=result['problem_size'],
            solver_used=result.get('solver_used'),
            routing_decision=result.get('routing_decision'),
            routing_reason=result.get('routing_reason'),
            routing_confidence=result.get('routing_confidence'),
            solution=result.get('solution'),
            cost=result.get('cost'),
            time_ms=result.get('time_ms'),
            energy_consumed_mj=result.get('energy_consumed_mj'),
            is_valid=result.get('is_valid'),
            solution_quality=result.get('solution_quality'),
            timestamp=result['timestamp'],
            error=result.get('error')
        )
        
    except ValueError as e:
        logger.error(f"Invalid TSP job request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TSP job execution failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/jobs/portfolio", response_model=JobResponse, tags=["Jobs"])
async def submit_portfolio_job(request: PortfolioJobRequest):
    """
    Submit a Portfolio Optimization problem for solving.
    
    Portfolio optimization seeks to select and weight assets to maximize return
    while minimizing risk. This involves:
    - Modern Portfolio Theory (Markowitz optimization)
    - Risk-return tradeoff analysis
    - Constraint handling (budget, sector limits)
    
    Applications include:
    - Investment portfolio management
    - Resource allocation
    - Project selection
    - Supply chain optimization
    
    **Process Flow**:
    1. Generate random portfolio with correlated asset returns
    2. Define risk-return objectives and constraints
    3. Route to solver (typically classical for convex problems)
    4. Solve for optimal weights
    5. Return allocation and expected metrics
    
    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/jobs/portfolio" \\
      -H "Content-Type: application/json" \\
      -d '{
        "num_assets": 20,
        "risk_aversion": 0.5,
        "edge_profile": "ground_server",
        "strategy": "quality_optimized"
      }'
    ```
    """
    try:
        logger.info(f"Received Portfolio job request: {request.dict()}")
        
        # Generate problem
        problem = PortfolioProblem(num_assets=request.num_assets)
        problem.generate(risk_aversion=request.risk_aversion, seed=request.seed)
        
        logger.info(f"Generated Portfolio problem: {request.num_assets} assets, "
                   f"risk_aversion={request.risk_aversion}")
        
        # Create orchestrator (database disabled for now - async driver issue)
        orchestrator = JobOrchestrator(
            enable_db=False,  # Disabled: async driver required
            db_url=None,
            strategy=request.strategy
        )
        
        # Execute job
        result = orchestrator.execute_job(
            problem=problem,
            edge_profile=request.edge_profile,
            strategy=request.strategy
        )
        
        logger.info(f"Portfolio job completed: {result['job_id']}, "
                   f"solver={result.get('solver_used')}, "
                   f"time={result.get('time_ms')}ms")
        
        # Convert to response model
        return JobResponse(
            job_id=result['job_id'],
            status="completed" if result['success'] else "failed",
            problem_type=result['problem_type'],
            problem_size=result['problem_size'],
            solver_used=result.get('solver_used'),
            routing_decision=result.get('routing_decision'),
            routing_reason=result.get('routing_reason'),
            routing_confidence=result.get('routing_confidence'),
            solution=result.get('solution'),
            cost=result.get('cost'),
            time_ms=result.get('time_ms'),
            energy_consumed_mj=result.get('energy_consumed_mj'),
            is_valid=result.get('is_valid'),
            solution_quality=result.get('solution_quality'),
            timestamp=result['timestamp'],
            error=result.get('error')
        )
        
    except ValueError as e:
        logger.error(f"Invalid Portfolio job request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Portfolio job execution failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/jobs/comparative", response_model=ComparativeJobResponse, tags=["Jobs"])
async def run_comparative_analysis(request: ComparativeJobRequest):
    """
    Run both classical and quantum solvers for side-by-side comparison.
    
    This endpoint executes the same problem with both solver types independently
    to enable direct performance comparison. Useful for:
    - Benchmarking quantum vs classical approaches
    - Understanding quantum advantage for specific problems
    - Demonstrating solver capabilities
    - Research and validation
    
    **Metrics Compared**:
    - Execution time (milliseconds)
    - Energy consumption (millijoules)
    - Solution quality
    - Resource utilization
    
    **Returns**:
    - Individual results for each solver
    - Comparative metrics (speedup, energy ratio)
    - Recommendation on which solver performed better
    - Detailed reasoning for recommendation
    
    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/jobs/comparative" \\
      -H "Content-Type: application/json" \\
      -d '{
        "problem_type": "maxcut",
        "problem_size": 25,
        "edge_profile": "aerospace",
        "seed": 42
      }'
    ```
    """
    try:
        logger.info(f"Received comparative analysis request: {request.dict()}")
        
        # Generate problem based on type
        if request.problem_type == "maxcut":
            problem = MaxCutProblem(num_nodes=request.problem_size)
            problem.generate(edge_probability=0.3, seed=request.seed)
        elif request.problem_type == "tsp":
            problem = TSPProblem(num_cities=request.problem_size)
            problem.generate(euclidean=True, seed=request.seed)
        elif request.problem_type == "portfolio":
            problem = PortfolioProblem(num_assets=request.problem_size)
            problem.generate(risk_aversion=0.5, seed=request.seed)
        else:
            raise ValueError(f"Unsupported problem type: {request.problem_type}")
        
        logger.info(f"Generated {request.problem_type} problem for comparative analysis: "
                   f"size={request.problem_size}")
        
        # Create orchestrator (database disabled for now - async driver issue)
        orchestrator = JobOrchestrator(
            enable_db=False,  # Disabled: async driver required
            db_url=None
        )
        
        # Execute comparative analysis
        result = orchestrator.execute_comparative(
            problem=problem,
            edge_profile=request.edge_profile
        )
        
        logger.info(f"Comparative analysis completed: {result['job_id']}, "
                   f"recommendation={result.get('recommendation')}")
        
        # Convert numpy types to Python native types
        result = convert_numpy_types(result)
        
        # Convert to response model
        return ComparativeJobResponse(
            job_id=result['job_id'],
            success=result['success'],
            problem_type=result['problem_type'],
            problem_size=result['problem_size'],
            classical=result['classical'],
            quantum=result['quantum'],
            speedup_factor=result.get('speedup_factor'),
            energy_ratio=result.get('energy_ratio'),
            quality_diff=result.get('quality_diff'),
            recommendation=result['recommendation'],
            recommendation_reason=result['recommendation_reason'],
            timestamp=result['timestamp']
        )
        
    except ValueError as e:
        logger.error(f"Invalid comparative analysis request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Comparative analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Routing Analysis Endpoints
# =============================================================================

@app.post("/api/v1/routing/analyze", response_model=RoutingAnalysisResponse, tags=["Routing"])
async def analyze_routing(request: RoutingAnalysisRequest):
    """
    Analyze routing decision without executing the problem.
    
    This endpoint performs routing analysis to determine which solver would be
    selected for a given problem, WITHOUT actually solving it. This is useful for:
    - Understanding routing decisions before committing resources
    - Capacity planning and resource estimation
    - Algorithm selection research
    - User education and transparency
    
    **Analysis Includes**:
    - Recommended solver (classical/quantum/hybrid)
    - Confidence score (0-1)
    - Detailed reasoning for decision
    - Performance estimates (time, energy)
    - Alternative solver options
    - Resource constraint analysis
    
    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/routing/analyze" \\
      -H "Content-Type: application/json" \\
      -d '{
        "problem_type": "maxcut",
        "problem_size": 30,
        "edge_profile": "aerospace",
        "strategy": "balanced"
      }'
    ```
    """
    try:
        logger.info(f"Received routing analysis request: {request.dict()}")
        
        # Generate problem for analysis (not solving)
        if request.problem_type == "maxcut":
            problem = MaxCutProblem(num_nodes=request.problem_size)
            problem.generate(edge_probability=0.3)
        elif request.problem_type == "tsp":
            problem = TSPProblem(num_cities=request.problem_size)
            problem.generate(euclidean=True)
        elif request.problem_type == "portfolio":
            problem = PortfolioProblem(num_assets=request.problem_size)
            problem.generate(risk_aversion=0.5)
        else:
            raise ValueError(f"Unsupported problem type: {request.problem_type}")
        
        # Create router with specified strategy
        strategy_map = {
            'balanced': RoutingStrategy.BALANCED,
            'energy_optimized': RoutingStrategy.ENERGY_OPTIMIZED,
            'latency_optimized': RoutingStrategy.LATENCY_OPTIMIZED,
            'quality_optimized': RoutingStrategy.QUALITY_OPTIMIZED
        }
        router = QuantumRouter(strategy=strategy_map[request.strategy])
        
        # Get edge environment
        profile_map = {
            'aerospace': DeploymentProfile.AEROSPACE,
            'mobile': DeploymentProfile.MOBILE,
            'ground_server': DeploymentProfile.GROUND_SERVER
        }
        edge_env = EdgeEnvironment(profile_map[request.edge_profile])
        
        # Perform routing analysis
        routing_result = router.route_problem(problem, edge_env)
        
        logger.info(f"Routing analysis complete: decision={routing_result['decision']}, "
                   f"confidence={routing_result['confidence']:.2f}")
        
        # Convert to response model
        return RoutingAnalysisResponse(**routing_result)
        
    except ValueError as e:
        logger.error(f"Invalid routing analysis request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Routing analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/routing/explain", tags=["Routing"])
async def explain_routing_decision(request: RoutingAnalysisRequest):
    """
    Get a detailed human-readable explanation of a routing decision.
    
    This endpoint provides a comprehensive, formatted explanation of why the
    router would make a particular decision for the given problem. The explanation
    includes:
    - Decision summary with confidence level
    - Detailed reasoning
    - Problem characteristics analysis
    - Resource constraint evaluation
    - Performance predictions
    - Alternative options
    
    The output is formatted for easy reading and can be used for:
    - User dashboards and reports
    - Documentation and transparency
    - Algorithm validation
    - Educational purposes
    
    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/routing/explain" \\
      -H "Content-Type: application/json" \\
      -d '{
        "problem_type": "maxcut",
        "problem_size": 30,
        "edge_profile": "aerospace",
        "strategy": "energy_optimized"
      }'
    ```
    """
    try:
        logger.info(f"Received routing explanation request: {request.dict()}")
        
        # Generate problem for analysis
        if request.problem_type == "maxcut":
            problem = MaxCutProblem(num_nodes=request.problem_size)
            problem.generate(edge_probability=0.3)
        elif request.problem_type == "tsp":
            problem = TSPProblem(num_cities=request.problem_size)
            problem.generate(euclidean=True)
        elif request.problem_type == "portfolio":
            problem = PortfolioProblem(num_assets=request.problem_size)
            problem.generate(risk_aversion=0.5)
        else:
            raise ValueError(f"Unsupported problem type: {request.problem_type}")
        
        # Create router
        strategy_map = {
            'balanced': RoutingStrategy.BALANCED,
            'energy_optimized': RoutingStrategy.ENERGY_OPTIMIZED,
            'latency_optimized': RoutingStrategy.LATENCY_OPTIMIZED,
            'quality_optimized': RoutingStrategy.QUALITY_OPTIMIZED
        }
        router = QuantumRouter(strategy=strategy_map[request.strategy])
        
        # Get edge environment
        profile_map = {
            'aerospace': DeploymentProfile.AEROSPACE,
            'mobile': DeploymentProfile.MOBILE,
            'ground_server': DeploymentProfile.GROUND_SERVER
        }
        edge_env = EdgeEnvironment(profile_map[request.edge_profile])
        
        # Get routing result
        routing_result = router.route_problem(problem, edge_env)
        
        # Generate explanation
        explanation = router.explain_decision(routing_result)
        
        logger.info(f"Routing explanation generated for {request.problem_type} problem")
        
        return {
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Invalid routing explanation request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Routing explanation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/routing/alternatives", tags=["Routing"])
async def suggest_routing_alternatives(request: RoutingAnalysisRequest):
    """
    Suggest alternative routing options with parameter adjustments.
    
    This endpoint analyzes a routing decision and suggests "what if" scenarios:
    - What if we increased power budget?
    - What if we allowed more execution time?
    - What if we used a different strategy?
    - What if we upgraded to a different edge profile?
    
    Each suggestion includes:
    - Type of adjustment needed
    - Expected improvements
    - Feasibility assessment
    - Implementation recommendations
    
    Useful for:
    - Capacity planning
    - Hardware upgrade decisions
    - Cost-benefit analysis
    - Performance optimization
    
    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/routing/alternatives" \\
      -H "Content-Type: application/json" \\
      -d '{
        "problem_type": "maxcut",
        "problem_size": 60,
        "edge_profile": "mobile",
        "strategy": "balanced"
      }'
    ```
    """
    try:
        logger.info(f"Received routing alternatives request: {request.dict()}")
        
        # Generate problem for analysis
        if request.problem_type == "maxcut":
            problem = MaxCutProblem(num_nodes=request.problem_size)
            problem.generate(edge_probability=0.3)
        elif request.problem_type == "tsp":
            problem = TSPProblem(num_cities=request.problem_size)
            problem.generate(euclidean=True)
        elif request.problem_type == "portfolio":
            problem = PortfolioProblem(num_assets=request.problem_size)
            problem.generate(risk_aversion=0.5)
        else:
            raise ValueError(f"Unsupported problem type: {request.problem_type}")
        
        # Create router
        strategy_map = {
            'balanced': RoutingStrategy.BALANCED,
            'energy_optimized': RoutingStrategy.ENERGY_OPTIMIZED,
            'latency_optimized': RoutingStrategy.LATENCY_OPTIMIZED,
            'quality_optimized': RoutingStrategy.QUALITY_OPTIMIZED
        }
        router = QuantumRouter(strategy=strategy_map[request.strategy])
        
        # Get edge environment
        profile_map = {
            'aerospace': DeploymentProfile.AEROSPACE,
            'mobile': DeploymentProfile.MOBILE,
            'ground_server': DeploymentProfile.GROUND_SERVER
        }
        edge_env = EdgeEnvironment(profile_map[request.edge_profile])
        
        # Get routing result
        routing_result = router.route_problem(problem, edge_env)
        
        # Generate suggestions
        suggestions = router.suggest_alternatives(routing_result)
        
        logger.info(f"Generated {len(suggestions)} routing alternatives")
        
        return {
            "current_decision": {
                "solver": routing_result['decision'],
                "confidence": routing_result['confidence'],
                "reasoning": routing_result['reasoning']
            },
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Invalid routing alternatives request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Routing alternatives generation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Configuration Endpoints
# =============================================================================

@app.get("/api/v1/config/edge-profiles", tags=["Configuration"])
async def get_edge_profiles():
    """
    Get information about available edge deployment profiles.
    
    Returns details for each supported edge profile:
    - **Aerospace**: Satellite/aircraft deployment (strict power limits)
    - **Mobile**: Smartphone/tablet (battery constrained)
    - **Ground Server**: Data center/server (relaxed constraints)
    
    Each profile includes:
    - Power budget (watts)
    - Thermal limits (Celsius)
    - Memory available (MB)
    - CPU cores
    - Maximum execution time (seconds)
    - Expected network latency (ms)
    
    **Example**:
    ```bash
    curl http://localhost:8000/api/v1/config/edge-profiles
    ```
    """
    try:
        profiles = settings.edge.profiles
        
        profile_info = {}
        for name, profile in profiles.items():
            profile_info[name] = EdgeProfileResponse(
                profile_name=name,
                power_budget_watts=profile.power_budget_watts,
                thermal_limit_celsius=profile.thermal_limit_celsius,
                memory_mb=profile.memory_mb,
                cpu_cores=profile.cpu_cores,
                max_execution_time_sec=profile.max_execution_time_sec,
                network_latency_ms=profile.network_latency_ms
            )
        
        return {
            "profiles": profile_info,
            "default_profile": settings.edge.default_profile
        }
        
    except Exception as e:
        logger.error(f"Failed to get edge profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/config/routing-strategies", tags=["Configuration"])
async def get_routing_strategies():
    """
    Get information about available routing strategies.
    
    Returns details for each supported routing strategy:
    
    - **Balanced**: Balance time, energy, and quality (40% time, 30% energy, 30% quality)
    - **Energy Optimized**: Minimize energy consumption (best for battery-powered)
    - **Latency Optimized**: Minimize execution time (best for real-time apps)
    - **Quality Optimized**: Maximize solution quality (best for critical decisions)
    
    Each strategy affects how the router scores classical vs quantum options
    and can override default routing logic.
    
    **Example**:
    ```bash
    curl http://localhost:8000/api/v1/config/routing-strategies
    ```
    """
    strategies = {
        "balanced": {
            "name": "Balanced",
            "description": "Balance time, energy, and quality using weighted scoring",
            "weights": {
                "time": 0.40,
                "energy": 0.30,
                "quality": 0.30
            },
            "use_cases": ["General purpose", "Mixed workloads", "Normal operations"],
            "priority": "Balanced performance across all metrics"
        },
        "energy_optimized": {
            "name": "Energy Optimized",
            "description": "Minimize total energy consumption",
            "priority": "Energy efficiency",
            "use_cases": [
                "Battery-powered edge devices",
                "Aerospace deployments",
                "Mobile applications",
                "Sustainable computing"
            ],
            "characteristics": "Prefers solver with lower energy/solution ratio"
        },
        "latency_optimized": {
            "name": "Latency Optimized",
            "description": "Minimize execution time",
            "priority": "Speed",
            "use_cases": [
                "Real-time applications",
                "Navigation systems",
                "Sensing and control",
                "Interactive systems"
            ],
            "characteristics": "May use more power to finish faster"
        },
        "quality_optimized": {
            "name": "Quality Optimized",
            "description": "Maximize solution quality",
            "priority": "Optimality",
            "use_cases": [
                "Critical decisions",
                "Mission planning",
                "Resource allocation",
                "High-stakes optimization"
            ],
            "characteristics": "Willing to spend more time and energy for better results"
        }
    }
    
    return {
        "strategies": strategies,
        "default_strategy": "balanced"
    }


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    API root endpoint with welcome message and quick links.
    
    Provides navigation to key API resources:
    - Interactive API documentation
    - Health check endpoint
    - System information
    - Quick start examples
    """
    return {
        "message": "Welcome to QuantumEdge Pipeline API",
        "version": settings.api.version,
        "documentation": {
            "swagger_ui": f"{settings.api.docs_url}",
            "redoc": f"{settings.api.redoc_url}",
            "openapi_schema": "/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "system_info": "/api/v1/system/info",
            "submit_maxcut": "/api/v1/jobs/maxcut",
            "submit_tsp": "/api/v1/jobs/tsp",
            "submit_portfolio": "/api/v1/jobs/portfolio",
            "comparative_analysis": "/api/v1/jobs/comparative",
            "routing_analysis": "/api/v1/routing/analyze",
            "edge_profiles": "/api/v1/config/edge-profiles",
            "routing_strategies": "/api/v1/config/routing-strategies"
        },
        "quick_start": {
            "example": "curl -X POST http://localhost:8000/api/v1/jobs/maxcut -H 'Content-Type: application/json' -d '{\"num_nodes\": 30, \"edge_probability\": 0.3}'",
            "documentation_url": "https://github.com/Gasta88/quantumedge-pipeline"
        },
        "support": {
            "github": "https://github.com/Gasta88/quantumedge-pipeline",
            "issues": "https://github.com/Gasta88/quantumedge-pipeline/issues"
        }
    }


# =============================================================================
# Application Lifespan Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Execute startup tasks when the application starts."""
    logger.info("=" * 80)
    logger.info("QuantumEdge Pipeline API Starting")
    logger.info("=" * 80)
    logger.info(f"Version: {settings.api.version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug Mode: {settings.api.debug}")
    logger.info(f"API Host: {settings.api.host}:{settings.api.port}")
    logger.info(f"Database: {'Enabled' if settings.database else 'Disabled'}")
    logger.info(f"Quantum Backend: {settings.quantum.backend}")
    logger.info(f"Documentation: http://{settings.api.host}:{settings.api.port}{settings.api.docs_url}")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Execute cleanup tasks when the application shuts down."""
    logger.info("=" * 80)
    logger.info("QuantumEdge Pipeline API Shutting Down")
    logger.info("=" * 80)
    # Add any cleanup logic here (close connections, save state, etc.)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.api.log_level.lower(),
        workers=settings.api.workers if not settings.api.debug else 1
    )
