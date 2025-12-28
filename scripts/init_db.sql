-- ==============================================================================
-- QuantumEdge Pipeline - Database Schema Initialization
-- ==============================================================================
-- This schema tracks quantum-classical hybrid optimization job executions,
-- enabling analysis of routing decisions, performance metrics, and quantum
-- advantage across different problem types and edge computing environments.
-- ==============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID generation
CREATE EXTENSION IF NOT EXISTS "timescaledb"; -- Time-series database capabilities

-- ==============================================================================
-- TABLE 1: problems
-- ==============================================================================
-- Purpose: Store optimization problem definitions and their characteristics.
-- This table contains the problem instances that are submitted for solving,
-- including the problem structure (graph data) and metadata.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS problems (
    -- Unique identifier for the problem instance
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Type of optimization problem
    problem_type VARCHAR(50) NOT NULL CHECK (problem_type IN ('maxcut', 'tsp', 'portfolio')),
    
    -- Size metric: number of nodes (MaxCut), cities (TSP), or assets (Portfolio)
    problem_size INTEGER NOT NULL CHECK (problem_size > 0),
    
    -- Problem structure stored as JSON (edges, weights, constraints, etc.)
    graph_data JSONB NOT NULL,
    
    -- When the problem was created/submitted
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Additional problem characteristics (complexity, sparsity, etc.)
    metadata JSONB DEFAULT '{}'::JSONB
);

-- Index for querying by problem type
CREATE INDEX idx_problems_type ON problems(problem_type);

-- Index for querying by problem size
CREATE INDEX idx_problems_size ON problems(problem_size);

-- Index for time-based queries
CREATE INDEX idx_problems_created_at ON problems(created_at DESC);

-- GIN index for efficient JSONB queries on metadata
CREATE INDEX idx_problems_metadata ON problems USING GIN(metadata);

-- Add helpful comment
COMMENT ON TABLE problems IS 'Stores optimization problem definitions including graph structure and metadata';
COMMENT ON COLUMN problems.problem_type IS 'Type of problem: maxcut (graph partitioning), tsp (traveling salesman), portfolio (asset optimization)';
COMMENT ON COLUMN problems.graph_data IS 'JSONB structure containing problem-specific data: edges, weights, constraints';
COMMENT ON COLUMN problems.metadata IS 'Additional characteristics: complexity score, sparsity ratio, structural properties';

-- ==============================================================================
-- TABLE 2: job_executions
-- ==============================================================================
-- Purpose: Track execution details for each quantum-classical routing decision.
-- Records the routing choice, execution performance, and resource utilization
-- for comparative analysis and optimization.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS job_executions (
    -- Unique identifier for this job execution
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Reference to the problem being solved
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    
    -- When the job was executed (used for time-series partitioning)
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Routing decision: which solver was selected
    routing_decision VARCHAR(20) NOT NULL CHECK (routing_decision IN ('classical', 'quantum', 'hybrid')),
    
    -- Explanation of why this routing decision was made
    routing_reason TEXT NOT NULL,
    
    -- Total execution time in milliseconds
    execution_time_ms INTEGER NOT NULL CHECK (execution_time_ms >= 0),
    
    -- Energy consumed in millijoules (for power efficiency analysis)
    energy_consumed_mj FLOAT NOT NULL CHECK (energy_consumed_mj >= 0),
    
    -- Quality of the solution (0.0 = worst, 1.0 = optimal)
    solution_quality FLOAT NOT NULL CHECK (solution_quality >= 0 AND solution_quality <= 1),
    
    -- Quantum advantage: speedup ratio compared to classical baseline
    -- Values > 1.0 indicate quantum advantage
    quantum_advantage_ratio FLOAT CHECK (quantum_advantage_ratio >= 0),
    
    -- Edge computing profile where job was executed
    edge_profile VARCHAR(50) NOT NULL CHECK (edge_profile IN ('aerospace', 'mobile', 'ground')),
    
    -- Percentage of available power budget used (0-100)
    power_budget_used FLOAT NOT NULL CHECK (power_budget_used >= 0 AND power_budget_used <= 100),
    
    -- Solver-specific details (circuit depth, iterations, convergence data, etc.)
    solver_metadata JSONB DEFAULT '{}'::JSONB
);

-- Index for problem lookups
CREATE INDEX idx_job_executions_problem_id ON job_executions(problem_id);

-- Index for timestamp-based queries (critical for time-series)
CREATE INDEX idx_job_executions_timestamp ON job_executions(timestamp DESC);

-- Index for filtering by routing decision
CREATE INDEX idx_job_executions_routing ON job_executions(routing_decision);

-- Index for filtering by edge profile
CREATE INDEX idx_job_executions_edge_profile ON job_executions(edge_profile);

-- Composite index for common filtering patterns
CREATE INDEX idx_job_executions_routing_profile ON job_executions(routing_decision, edge_profile, timestamp DESC);

-- GIN index for solver metadata queries
CREATE INDEX idx_job_executions_solver_metadata ON job_executions USING GIN(solver_metadata);

-- Add helpful comments
COMMENT ON TABLE job_executions IS 'Tracks each quantum-classical job execution with routing decision and performance metrics';
COMMENT ON COLUMN job_executions.routing_decision IS 'Solver selected: classical (traditional), quantum (QAOA/VQE), hybrid (combined approach)';
COMMENT ON COLUMN job_executions.routing_reason IS 'Human-readable explanation: problem size, quantum advantage estimate, resource constraints';
COMMENT ON COLUMN job_executions.quantum_advantage_ratio IS 'Speedup ratio: quantum_time / classical_time. Values > 1.0 show quantum advantage';
COMMENT ON COLUMN job_executions.edge_profile IS 'Computing environment: aerospace (strict power), mobile (battery), ground (relaxed constraints)';
COMMENT ON COLUMN job_executions.solver_metadata IS 'Solver details: circuit_depth, num_layers, optimizer_iterations, convergence_history';

-- Convert job_executions to TimescaleDB hypertable
-- Partition by timestamp with 1-day chunks for optimal query performance
SELECT create_hypertable(
    'job_executions',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ==============================================================================
-- TABLE 3: performance_metrics
-- ==============================================================================
-- Purpose: Store high-frequency time-series performance metrics during execution.
-- Captures detailed telemetry for monitoring, profiling, and analysis.
-- Examples: CPU usage, memory allocation, quantum gate execution time, etc.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    -- Timestamp of the metric measurement (primary key for time-series)
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Name of the metric being measured
    metric_name VARCHAR(100) NOT NULL,
    
    -- Numeric value of the metric
    metric_value FLOAT NOT NULL,
    
    -- Reference to the job execution (optional for system-wide metrics)
    job_id UUID REFERENCES job_executions(job_id) ON DELETE CASCADE,
    
    -- Additional tags for grouping/filtering (e.g., {"component": "quantum_simulator", "metric_type": "latency"})
    tags JSONB DEFAULT '{}'::JSONB
);

-- Index for time-based queries (most common access pattern)
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp DESC);

-- Index for metric name lookups
CREATE INDEX idx_performance_metrics_name ON performance_metrics(metric_name);

-- Index for job-specific metrics
CREATE INDEX idx_performance_metrics_job_id ON performance_metrics(job_id);

-- Composite index for filtered time-series queries
CREATE INDEX idx_performance_metrics_name_time ON performance_metrics(metric_name, timestamp DESC);

-- GIN index for tag-based filtering
CREATE INDEX idx_performance_metrics_tags ON performance_metrics USING GIN(tags);

-- Add helpful comments
COMMENT ON TABLE performance_metrics IS 'High-frequency time-series metrics for monitoring and profiling job executions';
COMMENT ON COLUMN performance_metrics.metric_name IS 'Examples: cpu_usage_percent, memory_mb, circuit_execution_time_ms, solver_iteration_time';
COMMENT ON COLUMN performance_metrics.tags IS 'Flexible tags for filtering: component, host, solver_type, metric_category';

-- Convert performance_metrics to TimescaleDB hypertable
-- Use smaller 1-hour chunks for high-frequency data
SELECT create_hypertable(
    'performance_metrics',
    'timestamp',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ==============================================================================
-- TimescaleDB Retention and Compression Policies
-- ==============================================================================
-- Automatically manage data lifecycle for optimal storage and performance
-- ==============================================================================

-- Compress job_executions older than 7 days (reduce storage by ~90%)
SELECT add_compression_policy('job_executions', INTERVAL '7 days', if_not_exists => TRUE);

-- Compress performance_metrics older than 1 day (high-frequency data)
SELECT add_compression_policy('performance_metrics', INTERVAL '1 day', if_not_exists => TRUE);

-- Retention policy: delete job_executions older than 1 year
SELECT add_retention_policy('job_executions', INTERVAL '1 year', if_not_exists => TRUE);

-- Retention policy: delete performance_metrics older than 90 days
SELECT add_retention_policy('performance_metrics', INTERVAL '90 days', if_not_exists => TRUE);

-- ==============================================================================
-- Continuous Aggregates (Pre-computed views for fast queries)
-- ==============================================================================
-- Create materialized views that automatically update for common analytics
-- ==============================================================================

-- Hourly aggregates: job counts, average execution times, and success rates
CREATE MATERIALIZED VIEW IF NOT EXISTS job_executions_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    routing_decision,
    edge_profile,
    COUNT(*) AS job_count,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    AVG(energy_consumed_mj) AS avg_energy_mj,
    AVG(solution_quality) AS avg_quality,
    AVG(quantum_advantage_ratio) AS avg_quantum_advantage,
    AVG(power_budget_used) AS avg_power_used
FROM job_executions
GROUP BY bucket, routing_decision, edge_profile
WITH NO DATA;

-- Refresh policy: update every hour
SELECT add_continuous_aggregate_policy('job_executions_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW job_executions_hourly IS 'Hourly aggregated statistics for dashboard and reporting';

-- ==============================================================================
-- Helper Views
-- ==============================================================================
-- Convenient views for common queries
-- ==============================================================================

-- View: Recent job summary with problem details
CREATE OR REPLACE VIEW recent_jobs_summary AS
SELECT
    je.job_id,
    je.timestamp,
    p.problem_type,
    p.problem_size,
    je.routing_decision,
    je.routing_reason,
    je.execution_time_ms,
    je.solution_quality,
    je.quantum_advantage_ratio,
    je.edge_profile,
    je.power_budget_used
FROM job_executions je
JOIN problems p ON je.problem_id = p.id
ORDER BY je.timestamp DESC
LIMIT 100;

COMMENT ON VIEW recent_jobs_summary IS 'Last 100 job executions with problem context for quick dashboard queries';

-- View: Quantum advantage statistics by problem type
CREATE OR REPLACE VIEW quantum_advantage_by_problem AS
SELECT
    p.problem_type,
    COUNT(*) AS total_jobs,
    AVG(je.quantum_advantage_ratio) AS avg_quantum_advantage,
    AVG(CASE WHEN je.routing_decision = 'quantum' THEN je.execution_time_ms END) AS avg_quantum_time_ms,
    AVG(CASE WHEN je.routing_decision = 'classical' THEN je.execution_time_ms END) AS avg_classical_time_ms,
    AVG(je.solution_quality) AS avg_solution_quality
FROM job_executions je
JOIN problems p ON je.problem_id = p.id
GROUP BY p.problem_type;

COMMENT ON VIEW quantum_advantage_by_problem IS 'Comparative analysis of quantum vs classical performance by problem type';

-- ==============================================================================
-- Grant Permissions (adjust as needed for your security model)
-- ==============================================================================

-- Grant appropriate permissions to application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO qe_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO qe_user;

-- ==============================================================================
-- Initialization Complete
-- ==============================================================================

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'QuantumEdge Pipeline database schema initialized successfully';
    RAISE NOTICE 'Created tables: problems, job_executions (hypertable), performance_metrics (hypertable)';
    RAISE NOTICE 'Created views: recent_jobs_summary, quantum_advantage_by_problem';
    RAISE NOTICE 'Created continuous aggregate: job_executions_hourly';
    RAISE NOTICE 'Configured compression and retention policies';
END $$;
