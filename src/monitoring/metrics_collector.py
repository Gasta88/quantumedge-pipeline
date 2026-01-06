"""
Metrics Collection System for QuantumEdge Pipeline.

This module provides comprehensive metrics collection and analysis capabilities for
tracking job execution performance, resource utilization, and system health. It acts
as a high-level interface to the database manager, providing convenient methods for
recording and querying metrics data.

Key Features:
------------
1. **Job Execution Recording**: Store complete job execution records with all metrics
2. **Performance Metrics**: Record time-series data for monitoring and profiling
3. **Historical Analysis**: Query and filter job history with flexible criteria
4. **Statistical Analysis**: Calculate aggregate statistics (mean, median, percentiles)
5. **Data Export**: Export metrics to CSV or JSON for external analysis

The metrics collector uses async database operations for high-performance data
ingestion and querying, essential for handling concurrent job submissions and
real-time monitoring dashboards.

Architecture:
------------
    ┌─────────────────────────────────────────┐
    │       MetricsCollector                  │
    ├─────────────────────────────────────────┤
    │                                         │
    │  • Job Execution Recording              │
    │  • Performance Metrics Storage          │
    │  • Historical Query & Filtering         │
    │  • Statistical Aggregation              │
    │  • Data Export (CSV/JSON)               │
    │                                         │
    └─────────────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │   DatabaseManager     │
         │   (Async Operations)  │
         └───────────────────────┘

Example Usage:
-------------
```python
from src.monitoring.metrics_collector import MetricsCollector
from datetime import datetime, timedelta

# Initialize metrics collector
collector = MetricsCollector(
    database_url="postgresql+asyncpg://user:pass@localhost/quantumedge"
)

# Record job execution
async with collector:
    await collector.record_job_execution({
        'job_id': job_id,
        'problem_type': 'maxcut',
        'problem_size': 50,
        'solver_used': 'quantum',
        'routing_reason': 'Problem size optimal for quantum',
        'execution_time_ms': 1500,
        'energy_consumed_mj': 42.5,
        'solution_quality': 0.92,
        'edge_profile': 'aerospace'
    })
    
    # Record performance metric
    await collector.record_performance_metric(
        name='cpu_usage',
        value=85.5,
        job_id=job_id,
        tags={'component': 'quantum_simulator', 'host': 'edge-1'}
    )
    
    # Query job history
    jobs = await collector.get_job_history(
        filters={
            'problem_type': 'maxcut',
            'solver_type': 'quantum',
            'time_range': (start_time, end_time)
        }
    )
    
    # Calculate aggregate statistics
    stats = await collector.calculate_aggregate_stats(
        metric_name='execution_time_ms',
        time_range=(start_time, end_time)
    )
    
    print(f"Mean: {stats['mean']:.2f}ms")
    print(f"Median: {stats['median']:.2f}ms")
    print(f"P99: {stats['p99']:.2f}ms")
    
    # Export metrics
    await collector.export_metrics(
        output_file='metrics_export.csv',
        format='csv'
    )
```

Design Principles:
-----------------
- **Async-first**: All operations are async for non-blocking I/O
- **Type-safe**: Strong typing with comprehensive validation
- **Observable**: Detailed logging for monitoring and debugging
- **Flexible**: Powerful filtering and aggregation capabilities
- **Export-friendly**: Easy data export for external analysis
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from uuid import UUID
import logging
import json
import csv
import statistics
from pathlib import Path

from src.monitoring.db_manager import DatabaseManager, DatabaseOperationError


# Configure module logger
logger = logging.getLogger(__name__)


class MetricsCollectionError(Exception):
    """Base exception for metrics collection errors."""
    pass


class ValidationError(MetricsCollectionError):
    """Raised when metric validation fails."""
    pass


class ExportError(MetricsCollectionError):
    """Raised when metric export fails."""
    pass


class MetricsCollector:
    """
    High-level metrics collection and analysis system for QuantumEdge Pipeline.
    
    This class provides convenient methods for recording job execution metrics,
    performance telemetry, and historical analysis. It wraps the DatabaseManager
    with additional validation, aggregation, and export capabilities.
    
    Attributes:
        db_manager (DatabaseManager): Database manager for async operations
        _metrics_buffer (List): Buffer for batched metric writes
        _buffer_size (int): Maximum buffer size before automatic flush
    
    Configuration:
        database_url (str): Database connection URL
        buffer_size (int): Metric buffer size for batch writes (default: 100)
        enable_buffering (bool): Enable metric buffering for efficiency
    
    Thread Safety:
        This class is not thread-safe. Create separate instances for
        concurrent operations or use external synchronization.
    """
    
    def __init__(
        self,
        database_url: str,
        buffer_size: int = 100,
        enable_buffering: bool = True,
        pool_size: int = 10,
        echo: bool = False
    ):
        """
        Initialize the metrics collector with database configuration.
        
        Args:
            database_url: Database connection URL (PostgreSQL with asyncpg)
            buffer_size: Number of metrics to buffer before batch write
            enable_buffering: Enable metric buffering for performance
            pool_size: Database connection pool size
            echo: Enable SQL query logging for debugging
        
        Raises:
            ValueError: If database_url is invalid
        
        Example:
            >>> collector = MetricsCollector(
            ...     database_url="postgresql+asyncpg://user:pass@localhost/quantumedge",
            ...     buffer_size=100,
            ...     enable_buffering=True
            ... )
        """
        logger.info("Initializing MetricsCollector")
        
        if not database_url:
            raise ValueError("database_url is required")
        
        # Initialize database manager
        self.db_manager = DatabaseManager(
            database_url=database_url,
            pool_size=pool_size,
            echo=echo
        )
        
        # Buffering configuration
        self._buffer_size = buffer_size
        self._enable_buffering = enable_buffering
        self._metrics_buffer: List[Tuple] = []
        
        # Statistics
        self._jobs_recorded = 0
        self._metrics_recorded = 0
        
        logger.info(f"MetricsCollector initialized (buffering={'enabled' if enable_buffering else 'disabled'})")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.db_manager.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - flush buffer and cleanup."""
        if self._enable_buffering:
            await self.flush_metrics_buffer()
        await self.db_manager.__aexit__(exc_type, exc_val, exc_tb)
    
    # =========================================================================
    # CORE METHODS - Job Execution Recording
    # =========================================================================
    
    async def record_job_execution(self, job_result: Dict[str, Any]) -> UUID:
        """
        Store complete job execution record in database.
        
        Records all relevant information about a job execution including:
        - Job identification and timing
        - Problem characteristics
        - Routing decision and reasoning
        - Execution metrics (time, energy, quality)
        - Resource utilization
        - Solver metadata
        
        This is the main method for recording job executions from the orchestrator.
        It validates input, extracts relevant fields, and stores them in the
        database for later analysis.
        
        Args:
            job_result: Complete job result dictionary containing:
                - job_id (str): Unique job identifier
                - timestamp (datetime): Execution timestamp
                - problem_type (str): Type of problem ('maxcut', 'tsp', 'portfolio')
                - problem_size (int): Problem size (nodes/cities/assets)
                - solver_used (str): Solver type ('classical', 'quantum', 'hybrid')
                - routing_decision (str): Same as solver_used
                - routing_reason (str): Explanation for routing choice
                - routing_confidence (float): Confidence score (0-1)
                - execution_time_ms (float): Total execution time
                - energy_consumed_mj (float): Energy consumed
                - solution_quality (float): Solution quality score (0-1)
                - edge_profile (str): Edge environment used
                - problem_analysis (Dict): Analysis results (optional)
                - solver_metadata (Dict): Solver-specific data (optional)
        
        Returns:
            UUID: The job execution ID stored in database
        
        Raises:
            ValidationError: If required fields are missing or invalid
            DatabaseOperationError: If database insertion fails
        
        Example:
            >>> job_result = {
            ...     'job_id': str(uuid4()),
            ...     'timestamp': datetime.now(),
            ...     'problem_type': 'maxcut',
            ...     'problem_size': 50,
            ...     'solver_used': 'quantum',
            ...     'routing_decision': 'quantum',
            ...     'routing_reason': 'Problem size optimal for quantum advantage',
            ...     'routing_confidence': 0.85,
            ...     'execution_time_ms': 1500.0,
            ...     'energy_consumed_mj': 42.5,
            ...     'solution_quality': 0.92,
            ...     'edge_profile': 'aerospace'
            ... }
            >>> job_id = await collector.record_job_execution(job_result)
        """
        logger.info(f"Recording job execution: {job_result.get('job_id', 'unknown')}")
        
        # Validate required fields
        required_fields = [
            'job_id', 'problem_type', 'problem_size', 'solver_used',
            'routing_decision', 'routing_reason', 'execution_time_ms',
            'energy_consumed_mj', 'solution_quality', 'edge_profile'
        ]
        
        missing_fields = [field for field in required_fields if field not in job_result]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")
        
        # Extract and validate fields
        try:
            job_id = job_result['job_id']
            if isinstance(job_id, str):
                job_id = UUID(job_id)
            
            timestamp = job_result.get('timestamp', datetime.utcnow())
            problem_type = job_result['problem_type']
            problem_size = int(job_result['problem_size'])
            routing_decision = job_result['routing_decision']
            routing_reason = job_result['routing_reason']
            execution_time_ms = int(job_result['execution_time_ms'])
            energy_consumed_mj = float(job_result['energy_consumed_mj'])
            solution_quality = float(job_result['solution_quality'])
            edge_profile = job_result['edge_profile']
            
            # Optional fields
            quantum_advantage_ratio = job_result.get('quantum_advantage_ratio')
            if quantum_advantage_ratio is not None:
                quantum_advantage_ratio = float(quantum_advantage_ratio)
            
            # Calculate power budget used (placeholder if not provided)
            power_budget_used = job_result.get('power_budget_used', 0.0)
            if power_budget_used == 0.0:
                # Estimate from energy and execution time
                # Assuming 100mW baseline power
                estimated_power_w = (energy_consumed_mj / execution_time_ms) if execution_time_ms > 0 else 0.1
                # Assume 1W budget for edge devices
                power_budget_used = min(100.0, (estimated_power_w / 1.0) * 100.0)
            
            # Solver metadata
            solver_metadata = job_result.get('solver_metadata', {})
            if not solver_metadata:
                solver_metadata = {
                    'solver_type': routing_decision,
                    'routing_confidence': job_result.get('routing_confidence', 0.0),
                    'problem_analysis': job_result.get('problem_analysis', {})
                }
            
        except (ValueError, KeyError, TypeError) as e:
            raise ValidationError(f"Field validation error: {e}") from e
        
        # First, ensure problem exists in database
        problem_id = await self._ensure_problem_exists(
            problem_type=problem_type,
            problem_size=problem_size,
            graph_data=job_result.get('graph_data', {}),
            metadata=job_result.get('problem_analysis', {})
        )
        
        # Insert job execution record
        try:
            execution_id = await self.db_manager.insert_job_execution(
                problem_id=problem_id,
                routing_decision=routing_decision,
                routing_reason=routing_reason,
                execution_time_ms=execution_time_ms,
                energy_consumed_mj=energy_consumed_mj,
                solution_quality=solution_quality,
                edge_profile=edge_profile,
                power_budget_used=power_budget_used,
                quantum_advantage_ratio=quantum_advantage_ratio,
                solver_metadata=solver_metadata,
                timestamp=timestamp
            )
            
            self._jobs_recorded += 1
            logger.info(f"Job execution recorded: {execution_id}")
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to record job execution: {e}")
            raise DatabaseOperationError(f"Failed to record job: {e}") from e
    
    async def record_performance_metric(
        self,
        name: str,
        value: float,
        job_id: Optional[Union[str, UUID]] = None,
        tags: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Store time-series performance metric.
        
        Records individual metric data points for monitoring and profiling.
        Metrics can be job-specific (tied to a job_id) or system-wide
        (job_id=None for infrastructure metrics).
        
        Common Metric Names:
        -------------------
        - 'cpu_usage_percent': CPU utilization (0-100)
        - 'memory_mb': Memory usage in megabytes
        - 'gpu_usage_percent': GPU utilization (0-100)
        - 'network_mbps': Network throughput
        - 'circuit_depth': Quantum circuit depth
        - 'gate_count': Number of quantum gates
        - 'iteration_time_ms': Per-iteration execution time
        - 'queue_depth': Job queue depth
        
        Args:
            name: Metric name (e.g., 'cpu_usage', 'memory_mb')
            value: Numeric metric value
            job_id: Associated job ID (optional, for system-wide metrics)
            tags: Metadata for filtering:
                 {'solver_type': 'quantum', 'problem_type': 'maxcut',
                  'component': 'simulator', 'host': 'edge-node-1'}
            timestamp: Metric timestamp (defaults to now)
        
        Raises:
            ValidationError: If metric name or value is invalid
        
        Example:
            >>> # Job-specific metric
            >>> await collector.record_performance_metric(
            ...     name='cpu_usage_percent',
            ...     value=87.5,
            ...     job_id=job_id,
            ...     tags={'component': 'quantum_simulator', 'host': 'edge-1'}
            ... )
            >>> 
            >>> # System-wide metric
            >>> await collector.record_performance_metric(
            ...     name='queue_depth',
            ...     value=5,
            ...     tags={'queue': 'quantum_jobs'}
            ... )
        """
        # Validate inputs
        if not name:
            raise ValidationError("Metric name cannot be empty")
        
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"Metric value must be numeric, got: {type(value)}")
        
        # Convert job_id to UUID if string
        if job_id is not None and isinstance(job_id, str):
            try:
                job_id = UUID(job_id)
            except ValueError:
                raise ValidationError(f"Invalid job_id format: {job_id}")
        
        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}
        
        # Add to buffer or insert directly
        if self._enable_buffering:
            self._metrics_buffer.append((name, value, job_id, tags, timestamp))
            
            # Auto-flush if buffer is full
            if len(self._metrics_buffer) >= self._buffer_size:
                await self.flush_metrics_buffer()
        else:
            # Direct insertion
            await self.db_manager.insert_performance_metric(
                metric_name=name,
                metric_value=value,
                job_id=job_id,
                tags=tags,
                timestamp=timestamp
            )
            self._metrics_recorded += 1
            logger.debug(f"Recorded metric: {name}={value}")
    
    async def flush_metrics_buffer(self) -> int:
        """
        Flush buffered metrics to database.
        
        Writes all buffered metrics in a single batch operation for efficiency.
        Automatically called when buffer is full or on context exit.
        
        Returns:
            int: Number of metrics flushed
        
        Example:
            >>> # Manually flush buffer
            >>> count = await collector.flush_metrics_buffer()
            >>> print(f"Flushed {count} metrics")
        """
        if not self._metrics_buffer:
            return 0
        
        count = len(self._metrics_buffer)
        logger.debug(f"Flushing {count} buffered metrics")
        
        try:
            await self.db_manager.insert_performance_metrics_bulk(self._metrics_buffer)
            self._metrics_recorded += count
            self._metrics_buffer.clear()
            logger.info(f"Flushed {count} metrics to database")
            return count
        except Exception as e:
            logger.error(f"Failed to flush metrics buffer: {e}")
            raise DatabaseOperationError(f"Metrics flush failed: {e}") from e
    
    # =========================================================================
    # QUERY METHODS - Historical Analysis
    # =========================================================================
    
    async def get_job_history(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query historical jobs with flexible filtering.
        
        Retrieves job execution records matching specified criteria.
        Supports filtering by time range, problem type, solver type,
        edge profile, and more.
        
        Filter Options:
        --------------
        - time_range: Tuple[datetime, datetime] - (start, end) time range
        - problem_type: str - Filter by problem type ('maxcut', 'tsp', 'portfolio')
        - solver_type: str - Filter by solver ('classical', 'quantum', 'hybrid')
        - edge_profile: str - Filter by edge environment
        - min_quality: float - Minimum solution quality (0-1)
        - max_execution_time_ms: int - Maximum execution time
        - min_problem_size: int - Minimum problem size
        - max_problem_size: int - Maximum problem size
        
        Args:
            filters: Dictionary of filter criteria (see above)
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
        
        Returns:
            List of job execution dictionaries, sorted by timestamp (newest first)
        
        Raises:
            ValidationError: If filter values are invalid
        
        Example:
            >>> # Get recent quantum jobs for MaxCut problems
            >>> jobs = await collector.get_job_history(
            ...     filters={
            ...         'problem_type': 'maxcut',
            ...         'solver_type': 'quantum',
            ...         'time_range': (
            ...             datetime.now() - timedelta(days=7),
            ...             datetime.now()
            ...         ),
            ...         'min_quality': 0.8
            ...     },
            ...     limit=50
            ... )
            >>> 
            >>> for job in jobs:
            ...     print(f"{job['timestamp']}: {job['problem_size']} nodes, "
            ...           f"quality={job['solution_quality']:.2%}")
        """
        filters = filters or {}
        logger.info(f"Querying job history with filters: {filters}")
        
        # Extract filter parameters
        time_range = filters.get('time_range')
        problem_type = filters.get('problem_type')
        solver_type = filters.get('solver_type')
        edge_profile = filters.get('edge_profile')
        min_quality = filters.get('min_quality')
        max_execution_time_ms = filters.get('max_execution_time_ms')
        min_problem_size = filters.get('min_problem_size')
        max_problem_size = filters.get('max_problem_size')
        
        # Use db_manager to get recent jobs with basic filters
        jobs = await self.db_manager.get_recent_jobs(
            limit=limit * 2,  # Get more to allow for additional filtering
            routing_decision=solver_type,
            edge_profile=edge_profile
        )
        
        # Apply additional filters
        filtered_jobs = []
        for job in jobs:
            # Time range filter
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= job['timestamp'] <= end_time):
                    continue
            
            # Problem type filter
            if problem_type and job['problem_type'] != problem_type:
                continue
            
            # Quality filter
            if min_quality is not None and job['solution_quality'] < min_quality:
                continue
            
            # Execution time filter
            if max_execution_time_ms is not None and job['execution_time_ms'] > max_execution_time_ms:
                continue
            
            # Problem size filters
            if min_problem_size is not None and job['problem_size'] < min_problem_size:
                continue
            if max_problem_size is not None and job['problem_size'] > max_problem_size:
                continue
            
            filtered_jobs.append(job)
            
            # Stop if we have enough results
            if len(filtered_jobs) >= limit + offset:
                break
        
        # Apply offset and limit
        result = filtered_jobs[offset:offset + limit]
        
        logger.info(f"Retrieved {len(result)} jobs from history")
        return result
    
    async def calculate_aggregate_stats(
        self,
        metric_name: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate aggregate statistics for a metric.
        
        Computes comprehensive statistical measures including:
        - Central tendency: mean, median
        - Dispersion: standard deviation, variance
        - Range: minimum, maximum
        - Percentiles: p50 (median), p90, p95, p99
        - Count: number of data points
        
        These statistics are useful for:
        - Performance analysis and trending
        - SLA monitoring (e.g., p99 latency)
        - Capacity planning
        - Anomaly detection
        
        Args:
            metric_name: Name of metric to analyze (e.g., 'execution_time_ms')
            time_range: Optional (start, end) datetime tuple
            filters: Additional filters (job_id, tags, etc.)
        
        Returns:
            Dictionary containing:
            {
                'count': int,           # Number of data points
                'mean': float,          # Average value
                'median': float,        # 50th percentile
                'std_dev': float,       # Standard deviation
                'variance': float,      # Variance
                'min': float,           # Minimum value
                'max': float,           # Maximum value
                'p50': float,           # 50th percentile (median)
                'p90': float,           # 90th percentile
                'p95': float,           # 95th percentile
                'p99': float,           # 99th percentile
                'range': float          # max - min
            }
        
        Raises:
            ValidationError: If metric_name is invalid or no data found
        
        Example:
            >>> # Analyze execution time for last 24 hours
            >>> end = datetime.now()
            >>> start = end - timedelta(hours=24)
            >>> 
            >>> stats = await collector.calculate_aggregate_stats(
            ...     metric_name='execution_time_ms',
            ...     time_range=(start, end)
            ... )
            >>> 
            >>> print(f"Mean: {stats['mean']:.2f}ms")
            >>> print(f"Median: {stats['median']:.2f}ms")
            >>> print(f"P99: {stats['p99']:.2f}ms")
            >>> print(f"Std Dev: {stats['std_dev']:.2f}ms")
        """
        logger.info(f"Calculating aggregate stats for metric: {metric_name}")
        
        if not metric_name:
            raise ValidationError("metric_name is required")
        
        # Set default time range if not provided (last 24 hours)
        if time_range is None:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            time_range = (start_time, end_time)
        
        start_time, end_time = time_range
        filters = filters or {}
        
        # Get metric data
        job_id = filters.get('job_id')
        tags = filters.get('tags')
        
        try:
            metrics = await self.db_manager.get_performance_metrics(
                metric_name=metric_name,
                start_time=start_time,
                end_time=end_time,
                job_id=job_id,
                tags=tags
            )
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            raise DatabaseOperationError(f"Failed to retrieve metrics: {e}") from e
        
        if not metrics:
            logger.warning(f"No data found for metric: {metric_name}")
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'variance': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p50': 0.0,
                'p90': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'range': 0.0
            }
        
        # Extract values
        values = [m['metric_value'] for m in metrics]
        
        # Calculate statistics
        count = len(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        
        # Standard deviation (need at least 2 points)
        std_dev = statistics.stdev(values) if count >= 2 else 0.0
        variance = statistics.variance(values) if count >= 2 else 0.0
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        # Calculate percentiles
        sorted_values = sorted(values)
        
        def percentile(data: List[float], p: float) -> float:
            """Calculate percentile value."""
            if not data:
                return 0.0
            k = (len(data) - 1) * (p / 100.0)
            f = int(k)
            c = f + 1 if f < len(data) - 1 else f
            d0 = data[f]
            d1 = data[c]
            return d0 + (d1 - d0) * (k - f)
        
        p50 = percentile(sorted_values, 50)
        p90 = percentile(sorted_values, 90)
        p95 = percentile(sorted_values, 95)
        p99 = percentile(sorted_values, 99)
        
        stats = {
            'count': count,
            'mean': mean_val,
            'median': median_val,
            'std_dev': std_dev,
            'variance': variance,
            'min': min_val,
            'max': max_val,
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'range': range_val
        }
        
        logger.info(f"Calculated stats for {metric_name}: count={count}, mean={mean_val:.2f}")
        return stats
    
    # =========================================================================
    # EXPORT METHODS - Data Export
    # =========================================================================
    
    async def export_metrics(
        self,
        output_file: Union[str, Path],
        format: str = 'csv',
        filters: Optional[Dict[str, Any]] = None,
        include_headers: bool = True
    ) -> int:
        """
        Export metrics data to file for external analysis.
        
        Exports job execution history or performance metrics to CSV or JSON
        format for analysis in external tools (Excel, R, Python notebooks, etc.).
        
        Supported Formats:
        -----------------
        - 'csv': Comma-separated values (Excel-compatible)
        - 'json': JSON array of objects
        
        Args:
            output_file: Output file path
            format: Export format ('csv' or 'json')
            filters: Query filters (same as get_job_history)
            include_headers: Include column headers (CSV only)
        
        Returns:
            int: Number of records exported
        
        Raises:
            ValueError: If format is invalid
            ExportError: If export fails
        
        Example:
            >>> # Export last week's jobs to CSV
            >>> count = await collector.export_metrics(
            ...     output_file='metrics_export.csv',
            ...     format='csv',
            ...     filters={
            ...         'time_range': (
            ...             datetime.now() - timedelta(days=7),
            ...             datetime.now()
            ...         )
            ...     }
            ... )
            >>> print(f"Exported {count} records")
            >>> 
            >>> # Export to JSON
            >>> count = await collector.export_metrics(
            ...     output_file='metrics.json',
            ...     format='json',
            ...     filters={'solver_type': 'quantum'}
            ... )
        """
        logger.info(f"Exporting metrics to {output_file} (format={format})")
        
        # Validate format
        if format not in ['csv', 'json']:
            raise ValueError(f"Invalid format: {format}. Must be 'csv' or 'json'")
        
        # Get data
        try:
            jobs = await self.get_job_history(filters=filters, limit=10000)
        except Exception as e:
            raise ExportError(f"Failed to retrieve data for export: {e}") from e
        
        if not jobs:
            logger.warning("No data to export")
            return 0
        
        # Export based on format
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                await self._export_csv(jobs, output_path, include_headers)
            elif format == 'json':
                await self._export_json(jobs, output_path)
            
            logger.info(f"Successfully exported {len(jobs)} records to {output_file}")
            return len(jobs)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise ExportError(f"Failed to export metrics: {e}") from e
    
    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    
    async def _ensure_problem_exists(
        self,
        problem_type: str,
        problem_size: int,
        graph_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> UUID:
        """
        Ensure problem exists in database, create if needed.
        
        Args:
            problem_type: Problem type
            problem_size: Problem size
            graph_data: Graph structure data
            metadata: Problem metadata
        
        Returns:
            UUID: Problem ID
        """
        # For now, always create new problem record
        # In production, might want to check for duplicates
        problem_id = await self.db_manager.insert_problem(
            problem_type=problem_type,
            problem_size=problem_size,
            graph_data=graph_data,
            metadata=metadata
        )
        return problem_id
    
    async def _export_csv(
        self,
        jobs: List[Dict[str, Any]],
        output_path: Path,
        include_headers: bool
    ) -> None:
        """Export jobs to CSV format."""
        with open(output_path, 'w', newline='') as csvfile:
            if not jobs:
                return
            
            fieldnames = list(jobs[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if include_headers:
                writer.writeheader()
            
            for job in jobs:
                # Convert non-serializable types
                row = {}
                for key, value in job.items():
                    if isinstance(value, (datetime, UUID)):
                        row[key] = str(value)
                    else:
                        row[key] = value
                writer.writerow(row)
    
    async def _export_json(
        self,
        jobs: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """Export jobs to JSON format."""
        # Convert non-serializable types
        serializable_jobs = []
        for job in jobs:
            serializable_job = {}
            for key, value in job.items():
                if isinstance(value, (datetime, UUID)):
                    serializable_job[key] = str(value)
                else:
                    serializable_job[key] = value
            serializable_jobs.append(serializable_job)
        
        with open(output_path, 'w') as jsonfile:
            json.dump(serializable_jobs, jsonfile, indent=2)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get collector statistics.
        
        Returns:
            Dictionary with statistics:
            {
                'jobs_recorded': int,
                'metrics_recorded': int,
                'buffer_size': int
            }
        """
        return {
            'jobs_recorded': self._jobs_recorded,
            'metrics_recorded': self._metrics_recorded,
            'buffer_size': len(self._metrics_buffer)
        }
    
    async def health_check(self) -> bool:
        """
        Check if metrics collector is healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            return await self.db_manager.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MetricsCollector(jobs_recorded={self._jobs_recorded}, "
                f"metrics_recorded={self._metrics_recorded}, "
                f"buffer_size={len(self._metrics_buffer)})")
