"""
Database Manager for QuantumEdge Pipeline.

This module provides async database operations for storing and retrieving
quantum-classical job execution data, performance metrics, and problem definitions.

Why Async?
----------
Async database operations are critical for handling concurrent job submissions
and monitoring without blocking the event loop. This allows the system to:
- Process multiple optimization jobs simultaneously
- Stream real-time metrics without latency
- Handle high-frequency performance data collection
- Scale to support multiple edge devices reporting concurrently

Example Usage:
--------------
```python
from src.monitoring.db_manager import DatabaseManager

# Initialize database manager
db = DatabaseManager(database_url="postgresql+asyncpg://user:pass@localhost/quantumedge")

# Store a problem and job execution
async with db:
    problem_id = await db.insert_problem(
        problem_type="maxcut",
        problem_size=50,
        graph_data={"edges": [[0, 1], [1, 2]], "weights": [1.0, 0.5]}
    )
    
    job_id = await db.insert_job_execution(
        problem_id=problem_id,
        routing_decision="quantum",
        routing_reason="Problem size optimal for quantum advantage",
        execution_time_ms=1250,
        energy_consumed_mj=45.2,
        solution_quality=0.95,
        edge_profile="aerospace"
    )
    
    # Query recent jobs
    recent_jobs = await db.get_recent_jobs(limit=10)
```
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

import logging

# Configure logger
logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class DatabaseOperationError(Exception):
    """Raised when a database operation fails."""
    pass


class DatabaseManager:
    """
    Async database manager for QuantumEdge Pipeline.
    
    Manages connections, transactions, and CRUD operations for tracking
    quantum-classical optimization jobs, performance metrics, and problem data.
    
    This class uses SQLAlchemy's async engine with asyncpg driver for
    non-blocking database operations, essential for handling concurrent
    job submissions and real-time monitoring.
    
    Attributes:
        engine (AsyncEngine): SQLAlchemy async engine with connection pooling
        session_factory (async_sessionmaker): Factory for creating async sessions
        _url (str): Database connection URL
    
    Example:
        >>> db = DatabaseManager("postgresql+asyncpg://user:pass@localhost/quantumedge")
        >>> async with db:
        ...     problem_id = await db.insert_problem("maxcut", 50, graph_data)
        ...     jobs = await db.get_recent_jobs(limit=5)
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        """
        Initialize the DatabaseManager with connection pool configuration.
        
        Args:
            database_url: Database connection URL in format:
                         postgresql+asyncpg://user:password@host:port/database
            pool_size: Number of persistent connections to maintain in the pool.
                      Default: 10 (suitable for moderate concurrent load)
            max_overflow: Maximum additional connections beyond pool_size.
                         Default: 20 (allows burst traffic handling)
            pool_timeout: Seconds to wait for available connection before timeout.
                         Default: 30.0 seconds
            pool_recycle: Seconds before recycling connections (prevents stale connections).
                         Default: 3600 (1 hour)
            echo: Enable SQL query logging for debugging.
                 Default: False (set True during development)
        
        Raises:
            DatabaseConnectionError: If database URL is invalid or connection fails
        
        Example:
            >>> db = DatabaseManager(
            ...     database_url="postgresql+asyncpg://qe_user:qe_pass@localhost/quantumedge",
            ...     pool_size=5,
            ...     echo=True  # Enable SQL logging
            ... )
        """
        self._url = database_url
        
        try:
            # Create async engine with connection pooling
            # Note: For async engines, SQLAlchemy automatically uses AsyncAdaptedQueuePool
            # We don't need to specify poolclass explicitly
            self.engine: AsyncEngine = create_async_engine(
                database_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                echo=echo,
                future=True,  # Use SQLAlchemy 2.0 style
            )
            
            # Create session factory for managing async sessions
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Keep objects usable after commit
            )
            
            logger.info(f"DatabaseManager initialized with pool_size={pool_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise DatabaseConnectionError(f"Database initialization failed: {e}")
    
    async def __aenter__(self):
        """Async context manager entry - verify connection on entry."""
        await self.health_check()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.close()
    
    async def close(self) -> None:
        """
        Close all database connections and dispose of the engine.
        
        Call this method when shutting down the application to ensure
        graceful cleanup of database resources.
        
        Example:
            >>> db = DatabaseManager(database_url)
            >>> # ... use database ...
            >>> await db.close()
        """
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    async def health_check(self) -> bool:
        """
        Verify database connection is healthy.
        
        Executes a simple query to ensure the database is reachable
        and responding. Useful for application startup checks and
        monitoring endpoints.
        
        Returns:
            bool: True if database is healthy, False otherwise
        
        Example:
            >>> db = DatabaseManager(database_url)
            >>> is_healthy = await db.health_check()
            >>> if is_healthy:
            ...     print("Database is ready")
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("SELECT 1"))
                result.scalar()
                logger.debug("Database health check passed")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def session(self):
        """
        Provide an async session with automatic transaction management.
        
        This context manager handles:
        - Session creation and cleanup
        - Automatic commit on success
        - Automatic rollback on error
        - Exception propagation with context
        
        Yields:
            AsyncSession: SQLAlchemy async session
        
        Raises:
            DatabaseOperationError: If transaction fails
        
        Example:
            >>> async with db.session() as session:
            ...     result = await session.execute(text("SELECT * FROM problems"))
            ...     # Automatically commits on success, rolls back on error
        """
        async_session = self.session_factory()
        try:
            yield async_session
            await async_session.commit()
        except SQLAlchemyError as e:
            await async_session.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise DatabaseOperationError(f"Transaction failed: {e}")
        finally:
            await async_session.close()
    
    # =========================================================================
    # Problem Management
    # =========================================================================
    
    async def insert_problem(
        self,
        problem_type: str,
        problem_size: int,
        graph_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """
        Insert a new optimization problem into the database.
        
        Stores problem definition including type, size, graph structure,
        and optional metadata for later analysis and job execution.
        
        Args:
            problem_type: Type of problem ('maxcut', 'tsp', 'portfolio')
            problem_size: Number of nodes/cities/assets
            graph_data: Problem structure as dict:
                       - For MaxCut: {"edges": [[0,1], [1,2]], "weights": [1.0, 0.5]}
                       - For TSP: {"cities": [...], "distances": [[...]]}
                       - For Portfolio: {"assets": [...], "correlations": [[...]]}
            metadata: Optional characteristics:
                     {"complexity": 0.75, "sparsity": 0.3, "source": "benchmark"}
        
        Returns:
            UUID: Unique identifier for the created problem
        
        Raises:
            DatabaseOperationError: If insertion fails
            ValueError: If problem_type is invalid
        
        Example:
            >>> problem_id = await db.insert_problem(
            ...     problem_type="maxcut",
            ...     problem_size=20,
            ...     graph_data={"edges": [[0, 1], [1, 2]], "weights": [1.0, 0.8]},
            ...     metadata={"complexity": 0.65, "sparsity": 0.4}
            ... )
        """
        valid_types = ["maxcut", "tsp", "portfolio"]
        if problem_type not in valid_types:
            raise ValueError(f"Invalid problem_type. Must be one of {valid_types}")
        
        if problem_size <= 0:
            raise ValueError("problem_size must be positive")
        
        problem_id = uuid4()
        metadata = metadata or {}
        
        query = text("""
            INSERT INTO problems (id, problem_type, problem_size, graph_data, metadata)
            VALUES (:id, :problem_type, :problem_size, :graph_data, :metadata)
            RETURNING id
        """)
        
        async with self.session() as session:
            result = await session.execute(
                query,
                {
                    "id": problem_id,
                    "problem_type": problem_type,
                    "problem_size": problem_size,
                    "graph_data": graph_data,
                    "metadata": metadata,
                },
            )
            created_id = result.scalar_one()
            logger.info(f"Inserted problem {created_id}: {problem_type} (size={problem_size})")
            return created_id
    
    async def get_problem(self, problem_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a problem by its UUID.
        
        Args:
            problem_id: UUID of the problem to retrieve
        
        Returns:
            Dict containing problem data, or None if not found
        
        Example:
            >>> problem = await db.get_problem(problem_id)
            >>> if problem:
            ...     print(f"Problem type: {problem['problem_type']}")
        """
        query = text("""
            SELECT id, problem_type, problem_size, graph_data, created_at, metadata
            FROM problems
            WHERE id = :problem_id
        """)
        
        async with self.session() as session:
            result = await session.execute(query, {"problem_id": problem_id})
            row = result.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "problem_type": row[1],
                    "problem_size": row[2],
                    "graph_data": row[3],
                    "created_at": row[4],
                    "metadata": row[5],
                }
            return None
    
    # =========================================================================
    # Job Execution Management
    # =========================================================================
    
    async def insert_job_execution(
        self,
        problem_id: UUID,
        routing_decision: str,
        routing_reason: str,
        execution_time_ms: int,
        energy_consumed_mj: float,
        solution_quality: float,
        edge_profile: str,
        power_budget_used: float,
        quantum_advantage_ratio: Optional[float] = None,
        solver_metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> UUID:
        """
        Insert a job execution record.
        
        Records the complete execution details including routing decision,
        performance metrics, and resource utilization for a quantum-classical
        optimization job.
        
        Args:
            problem_id: UUID of the problem being solved
            routing_decision: 'classical', 'quantum', or 'hybrid'
            routing_reason: Explanation for routing choice
                           e.g., "Problem size (50 nodes) optimal for quantum advantage"
            execution_time_ms: Total execution time in milliseconds
            energy_consumed_mj: Energy used in millijoules
            solution_quality: Solution quality score (0.0 to 1.0)
            edge_profile: Computing environment ('aerospace', 'mobile', 'ground_server')
            power_budget_used: Percentage of power budget consumed (0-100)
            quantum_advantage_ratio: Speedup vs classical baseline (optional)
            solver_metadata: Solver-specific details (optional):
                           {"circuit_depth": 10, "iterations": 100, "optimizer": "COBYLA"}
            timestamp: Execution timestamp (defaults to now)
        
        Returns:
            UUID: Unique identifier for the job execution
        
        Raises:
            DatabaseOperationError: If insertion fails
            ValueError: If parameters are invalid
        
        Example:
            >>> job_id = await db.insert_job_execution(
            ...     problem_id=problem_id,
            ...     routing_decision="quantum",
            ...     routing_reason="50 nodes ideal for QAOA quantum advantage",
            ...     execution_time_ms=1500,
            ...     energy_consumed_mj=42.5,
            ...     solution_quality=0.92,
            ...     edge_profile="aerospace",
            ...     power_budget_used=35.8,
            ...     quantum_advantage_ratio=2.3,
            ...     solver_metadata={"circuit_depth": 8, "optimizer": "COBYLA"}
            ... )
        """
        valid_decisions = ["classical", "quantum", "hybrid"]
        if routing_decision not in valid_decisions:
            raise ValueError(f"Invalid routing_decision. Must be one of {valid_decisions}")
        
        valid_profiles = ["aerospace", "mobile", "ground_server"]
        if edge_profile not in valid_profiles:
            raise ValueError(f"Invalid edge_profile. Must be one of {valid_profiles}")
        
        if not (0.0 <= solution_quality <= 1.0):
            raise ValueError("solution_quality must be between 0.0 and 1.0")
        
        if not (0.0 <= power_budget_used <= 100.0):
            raise ValueError("power_budget_used must be between 0.0 and 100.0")
        
        job_id = uuid4()
        timestamp = timestamp or datetime.utcnow()
        solver_metadata = solver_metadata or {}
        
        query = text("""
            INSERT INTO job_executions (
                job_id, problem_id, timestamp, routing_decision, routing_reason,
                execution_time_ms, energy_consumed_mj, solution_quality,
                quantum_advantage_ratio, edge_profile, power_budget_used, solver_metadata
            )
            VALUES (
                :job_id, :problem_id, :timestamp, :routing_decision, :routing_reason,
                :execution_time_ms, :energy_consumed_mj, :solution_quality,
                :quantum_advantage_ratio, :edge_profile, :power_budget_used, :solver_metadata
            )
            RETURNING job_id
        """)
        
        async with self.session() as session:
            result = await session.execute(
                query,
                {
                    "job_id": job_id,
                    "problem_id": problem_id,
                    "timestamp": timestamp,
                    "routing_decision": routing_decision,
                    "routing_reason": routing_reason,
                    "execution_time_ms": execution_time_ms,
                    "energy_consumed_mj": energy_consumed_mj,
                    "solution_quality": solution_quality,
                    "quantum_advantage_ratio": quantum_advantage_ratio,
                    "edge_profile": edge_profile,
                    "power_budget_used": power_budget_used,
                    "solver_metadata": solver_metadata,
                },
            )
            created_id = result.scalar_one()
            logger.info(
                f"Inserted job {created_id}: {routing_decision} "
                f"(time={execution_time_ms}ms, quality={solution_quality})"
            )
            return created_id
    
    async def get_recent_jobs(
        self,
        limit: int = 100,
        routing_decision: Optional[str] = None,
        edge_profile: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent job executions with optional filtering.
        
        Fetches the most recent job executions, useful for dashboard displays
        and recent performance analysis. Results include problem context.
        
        Args:
            limit: Maximum number of jobs to return (default: 100)
            routing_decision: Filter by decision type ('classical', 'quantum', 'hybrid')
            edge_profile: Filter by edge environment ('aerospace', 'mobile', 'ground_server')
        
        Returns:
            List of job execution dictionaries, sorted by timestamp (newest first)
        
        Example:
            >>> # Get last 10 quantum jobs on aerospace profile
            >>> jobs = await db.get_recent_jobs(
            ...     limit=10,
            ...     routing_decision="quantum",
            ...     edge_profile="aerospace"
            ... )
            >>> for job in jobs:
            ...     print(f"{job['timestamp']}: {job['problem_type']} -> {job['solution_quality']}")
        """
        conditions = []
        params = {"limit": limit}
        
        if routing_decision:
            conditions.append("je.routing_decision = :routing_decision")
            params["routing_decision"] = routing_decision
        
        if edge_profile:
            conditions.append("je.edge_profile = :edge_profile")
            params["edge_profile"] = edge_profile
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = text(f"""
            SELECT
                je.job_id,
                je.timestamp,
                je.routing_decision,
                je.routing_reason,
                je.execution_time_ms,
                je.energy_consumed_mj,
                je.solution_quality,
                je.quantum_advantage_ratio,
                je.edge_profile,
                je.power_budget_used,
                p.problem_type,
                p.problem_size
            FROM job_executions je
            JOIN problems p ON je.problem_id = p.id
            {where_clause}
            ORDER BY je.timestamp DESC
            LIMIT :limit
        """)
        
        async with self.session() as session:
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            return [
                {
                    "job_id": row[0],
                    "timestamp": row[1],
                    "routing_decision": row[2],
                    "routing_reason": row[3],
                    "execution_time_ms": row[4],
                    "energy_consumed_mj": row[5],
                    "solution_quality": row[6],
                    "quantum_advantage_ratio": row[7],
                    "edge_profile": row[8],
                    "power_budget_used": row[9],
                    "problem_type": row[10],
                    "problem_size": row[11],
                }
                for row in rows
            ]
    
    # =========================================================================
    # Performance Metrics Management
    # =========================================================================
    
    async def insert_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        job_id: Optional[UUID] = None,
        tags: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Insert a performance metric data point.
        
        Records high-frequency telemetry data for monitoring and profiling.
        Metrics can be job-specific or system-wide.
        
        Args:
            metric_name: Name of the metric being measured
                        e.g., 'cpu_usage_percent', 'memory_mb', 'circuit_execution_ms'
            metric_value: Numeric value of the metric
            job_id: Associated job UUID (optional for system metrics)
            tags: Additional metadata for filtering:
                 {"component": "quantum_simulator", "host": "edge-node-1"}
            timestamp: Measurement timestamp (defaults to now)
        
        Raises:
            DatabaseOperationError: If insertion fails
        
        Example:
            >>> # Record CPU usage during job execution
            >>> await db.insert_performance_metric(
            ...     metric_name="cpu_usage_percent",
            ...     metric_value=87.5,
            ...     job_id=job_id,
            ...     tags={"component": "quantum_simulator", "host": "edge-node-1"}
            ... )
        """
        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}
        
        query = text("""
            INSERT INTO performance_metrics (timestamp, metric_name, metric_value, job_id, tags)
            VALUES (:timestamp, :metric_name, :metric_value, :job_id, :tags)
        """)
        
        async with self.session() as session:
            await session.execute(
                query,
                {
                    "timestamp": timestamp,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "job_id": job_id,
                    "tags": tags,
                },
            )
            logger.debug(f"Inserted metric: {metric_name}={metric_value}")
    
    async def insert_performance_metrics_bulk(
        self,
        metrics: List[Tuple[str, float, Optional[UUID], Optional[Dict[str, Any]], Optional[datetime]]],
    ) -> None:
        """
        Bulk insert performance metrics for efficiency.
        
        When collecting high-frequency metrics, bulk insertion reduces
        database round-trips and improves throughput significantly.
        
        Args:
            metrics: List of tuples: (metric_name, metric_value, job_id, tags, timestamp)
        
        Example:
            >>> metrics = [
            ...     ("cpu_usage", 85.0, job_id, {"host": "node1"}, datetime.utcnow()),
            ...     ("memory_mb", 2048.5, job_id, {"host": "node1"}, datetime.utcnow()),
            ...     ("latency_ms", 12.3, job_id, {"host": "node1"}, datetime.utcnow()),
            ... ]
            >>> await db.insert_performance_metrics_bulk(metrics)
        """
        query = text("""
            INSERT INTO performance_metrics (timestamp, metric_name, metric_value, job_id, tags)
            VALUES (:timestamp, :metric_name, :metric_value, :job_id, :tags)
        """)
        
        params_list = [
            {
                "timestamp": timestamp or datetime.utcnow(),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "job_id": job_id,
                "tags": tags or {},
            }
            for metric_name, metric_value, job_id, tags, timestamp in metrics
        ]
        
        async with self.session() as session:
            await session.execute(query, params_list)
            logger.debug(f"Bulk inserted {len(metrics)} metrics")
    
    async def get_performance_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        job_id: Optional[UUID] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query performance metrics for a time range.
        
        Retrieves time-series metric data for analysis, visualization,
        and performance profiling.
        
        Args:
            metric_name: Name of the metric to query
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            job_id: Filter by specific job (optional)
            tags: Filter by tags (optional)
        
        Returns:
            List of metric data points with timestamps
        
        Example:
            >>> # Get CPU usage for last hour
            >>> end = datetime.utcnow()
            >>> start = end - timedelta(hours=1)
            >>> metrics = await db.get_performance_metrics(
            ...     metric_name="cpu_usage_percent",
            ...     start_time=start,
            ...     end_time=end,
            ...     tags={"component": "quantum_simulator"}
            ... )
            >>> avg_cpu = sum(m['metric_value'] for m in metrics) / len(metrics)
        """
        conditions = [
            "metric_name = :metric_name",
            "timestamp BETWEEN :start_time AND :end_time",
        ]
        params = {
            "metric_name": metric_name,
            "start_time": start_time,
            "end_time": end_time,
        }
        
        if job_id:
            conditions.append("job_id = :job_id")
            params["job_id"] = job_id
        
        if tags:
            conditions.append("tags @> :tags::jsonb")
            params["tags"] = tags
        
        where_clause = " AND ".join(conditions)
        
        query = text(f"""
            SELECT timestamp, metric_name, metric_value, job_id, tags
            FROM performance_metrics
            WHERE {where_clause}
            ORDER BY timestamp ASC
        """)
        
        async with self.session() as session:
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            return [
                {
                    "timestamp": row[0],
                    "metric_name": row[1],
                    "metric_value": row[2],
                    "job_id": row[3],
                    "tags": row[4],
                }
                for row in rows
            ]
    
    # =========================================================================
    # Analytics and Reporting
    # =========================================================================
    
    async def get_quantum_advantage_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get quantum advantage statistics by problem type.
        
        Analyzes comparative performance between quantum and classical
        solvers across different problem types.
        
        Args:
            start_time: Filter jobs after this time (optional)
            end_time: Filter jobs before this time (optional)
        
        Returns:
            List of statistics per problem type including:
            - Average quantum advantage ratio
            - Average execution times (quantum vs classical)
            - Average solution quality
            - Job counts
        
        Example:
            >>> stats = await db.get_quantum_advantage_stats()
            >>> for stat in stats:
            ...     print(f"{stat['problem_type']}: {stat['avg_quantum_advantage']:.2f}x advantage")
        """
        conditions = []
        params = {}
        
        if start_time:
            conditions.append("je.timestamp >= :start_time")
            params["start_time"] = start_time
        
        if end_time:
            conditions.append("je.timestamp <= :end_time")
            params["end_time"] = end_time
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = text(f"""
            SELECT
                p.problem_type,
                COUNT(*) as total_jobs,
                AVG(je.quantum_advantage_ratio) as avg_quantum_advantage,
                AVG(CASE WHEN je.routing_decision = 'quantum' THEN je.execution_time_ms END) as avg_quantum_time_ms,
                AVG(CASE WHEN je.routing_decision = 'classical' THEN je.execution_time_ms END) as avg_classical_time_ms,
                AVG(je.solution_quality) as avg_solution_quality,
                AVG(je.energy_consumed_mj) as avg_energy_mj
            FROM job_executions je
            JOIN problems p ON je.problem_id = p.id
            {where_clause}
            GROUP BY p.problem_type
            ORDER BY avg_quantum_advantage DESC
        """)
        
        async with self.session() as session:
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            return [
                {
                    "problem_type": row[0],
                    "total_jobs": row[1],
                    "avg_quantum_advantage": float(row[2]) if row[2] else None,
                    "avg_quantum_time_ms": float(row[3]) if row[3] else None,
                    "avg_classical_time_ms": float(row[4]) if row[4] else None,
                    "avg_solution_quality": float(row[5]) if row[5] else None,
                    "avg_energy_mj": float(row[6]) if row[6] else None,
                }
                for row in rows
            ]
    
    async def get_edge_profile_stats(
        self,
        edge_profile: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific edge computing profile.
        
        Analyzes performance characteristics and resource utilization
        for jobs executed on a particular edge profile.
        
        Args:
            edge_profile: Edge environment ('aerospace', 'mobile', 'ground_server')
            start_time: Filter jobs after this time (optional)
            end_time: Filter jobs before this time (optional)
        
        Returns:
            Statistics dictionary including averages and counts
        
        Example:
            >>> stats = await db.get_edge_profile_stats("aerospace")
            >>> print(f"Avg power usage: {stats['avg_power_budget_used']:.1f}%")
        """
        conditions = ["edge_profile = :edge_profile"]
        params = {"edge_profile": edge_profile}
        
        if start_time:
            conditions.append("timestamp >= :start_time")
            params["start_time"] = start_time
        
        if end_time:
            conditions.append("timestamp <= :end_time")
            params["end_time"] = end_time
        
        where_clause = " AND ".join(conditions)
        
        query = text(f"""
            SELECT
                COUNT(*) as total_jobs,
                AVG(execution_time_ms) as avg_execution_time_ms,
                AVG(energy_consumed_mj) as avg_energy_mj,
                AVG(solution_quality) as avg_solution_quality,
                AVG(power_budget_used) as avg_power_budget_used,
                COUNT(CASE WHEN routing_decision = 'quantum' THEN 1 END) as quantum_jobs,
                COUNT(CASE WHEN routing_decision = 'classical' THEN 1 END) as classical_jobs,
                COUNT(CASE WHEN routing_decision = 'hybrid' THEN 1 END) as hybrid_jobs
            FROM job_executions
            WHERE {where_clause}
        """)
        
        async with self.session() as session:
            result = await session.execute(query, params)
            row = result.fetchone()
            
            if row:
                return {
                    "edge_profile": edge_profile,
                    "total_jobs": row[0],
                    "avg_execution_time_ms": float(row[1]) if row[1] else 0.0,
                    "avg_energy_mj": float(row[2]) if row[2] else 0.0,
                    "avg_solution_quality": float(row[3]) if row[3] else 0.0,
                    "avg_power_budget_used": float(row[4]) if row[4] else 0.0,
                    "quantum_jobs": row[5],
                    "classical_jobs": row[6],
                    "hybrid_jobs": row[7],
                }
            return {}
