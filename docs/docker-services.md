# Docker Compose Services Documentation

## Services Overview

### 1. postgres-timescale
**Image:** `timescale/timescaledb:latest-pg15`

TimescaleDB is an open-source time-series database built on PostgreSQL. Perfect for storing performance metrics, execution logs, and time-series optimization results.

- **Port:** 5432
- **Database:** quantumedge
- **User:** qe_user
- **Password:** qe_pass (change in production!)
- **Data Volume:** `./data/postgres` (persistent storage)
- **Features:**
  - Time-series data optimization
  - Automatic data partitioning
  - Compression for historical data
  - Full PostgreSQL compatibility
  - Pre-configured hypertables for job executions and performance metrics

### 2. app (FastAPI Backend)
**Built from:** `Dockerfile`

The main FastAPI application serving the quantum-classical routing API.

- **Port:** 8000
- **Base Path:** `/app`
- **Source Volume:** `./src` → `/app/src` (hot reload)
- **Features:**
  - Automatic API documentation at `/docs`
  - Health endpoint at `/health`
  - Hot reload in development mode

**Dependencies:**
- postgres-timescale (with health check)

### 3. dashboard (Streamlit)
**Built from:** `Dockerfile`

Interactive Streamlit dashboard for visualization and control.

- **Port:** 8501
- **Source Volume:** `./dashboard` → `/app/dashboard`
- **Features:**
  - Real-time problem analysis
  - Routing decision visualization
  - Performance comparisons
  - Interactive problem submission

**Dependencies:**
- app (FastAPI backend)

## Network Configuration

**Network:** `quantumedge-network`
- **Type:** Bridge
- **Subnet:** 172.25.0.0/16
- All services communicate on this isolated network

## Volume Management

### Persistent Volumes
- `postgres_data` - TimescaleDB data (named volume)

### Development Volumes
- `./src` - Source code hot reload
- `./dashboard` - Dashboard hot reload
- `./logs` - Application logs

## Health Checks

All services have configured health checks with the following parameters:

| Service | Endpoint | Interval | Timeout | Retries | Start Period |
|---------|----------|----------|---------|---------|--------------|
| postgres-timescale | pg_isready | 10s | 5s | 5 | 30s |
| app | /health | 30s | 10s | 3 | 40s |
| dashboard | /_stcore/health | 30s | 10s | 3 | 40s |

## Quick Start Commands

### Start all services
```bash
docker-compose up -d
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
```

### Check service status
```bash
docker-compose ps
```

### Stop all services
```bash
docker-compose down
```

### Stop and remove volumes
```bash
docker-compose down -v
```

### Rebuild and restart
```bash
docker-compose up -d --build
```

### Execute commands in containers
```bash
# Database shell
docker-compose exec postgres-timescale psql -U qe_user -d quantumedge

# App shell
docker-compose exec app bash
```

## Service URLs

Once all services are running:

- **FastAPI Docs:** http://localhost:8000/docs
- **Streamlit Dashboard:** http://localhost:8501

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
nano .env
```

**Important variables:**
- `POSTGRES_PASSWORD` - Change for production!
- `SECRET_KEY` - Generate secure key for production
- `QUANTUM_BACKEND` - Choose: simulator, ibm_quantum, aws_braket

## Production Considerations

### Security
1. Change all default passwords
2. Use Docker secrets for sensitive data
3. Enable SSL/TLS for external connections
4. Restrict network access with firewall rules

### Performance
1. Tune PostgreSQL `shared_buffers` and `work_mem`
2. Scale app containers with `docker-compose up -d --scale app=3`
3. Use connection pooling (PgBouncer) for database

### Monitoring
2. Enable log aggregation (ELK/Loki)
3. Monitor resource usage with cAdvisor

### Backup
```bash
# Backup TimescaleDB
docker-compose exec postgres-timescale pg_dump -U qe_user quantumedge > backup.sql

# Restore database
docker-compose exec -T postgres-timescale psql -U qe_user quantumedge < backup.sql
```

## Troubleshooting

### Service won't start
```bash
# Check logs
docker-compose logs [service-name]

# Verify health
docker-compose ps
```

### Database connection issues
```bash
# Test connection
docker-compose exec app python -c "from src.config import config; print(config.database_url)"

# Check PostgreSQL logs
docker-compose logs postgres-timescale
```

### Port conflicts
If ports are already in use, modify in `.env`:
```bash
API_PORT=8001
DASHBOARD_PORT=8502
```

## TimescaleDB Specific Features

### Create hypertable for time-series metrics
```sql
-- Connect to database
docker-compose exec postgres-timescale psql -U qe_user -d quantumedge

-- Create hypertable
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    problem_id INTEGER,
    solver_type TEXT,
    execution_time DOUBLE PRECISION,
    memory_mb DOUBLE PRECISION
);

SELECT create_hypertable('metrics', 'time');
```

### Data retention policy
```sql
-- Keep only 30 days of detailed data
SELECT add_retention_policy('metrics', INTERVAL '30 days');

-- Compress data older than 7 days
SELECT add_compression_policy('metrics', INTERVAL '7 days');
```
