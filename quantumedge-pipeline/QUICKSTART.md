# Quick Start Guide

## Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

## Initial Setup

### 1. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env
```

**Important: Update these values in .env:**
- `POSTGRES_PASSWORD` - Change from default!
- `GRAFANA_PASSWORD` - Change from default!
- `SECRET_KEY` - Generate a secure random key

### 2. Create Required Directories
```bash
make dev-setup
```

This creates:
- `./data/postgres` - Database storage
- `./data/grafana` - Grafana dashboards
- `./logs` - Application logs
- `./backups` - Database backups

### 3. Build and Start Services
```bash
# Build Docker images
make build

# Start all services
make up
```

### 4. Verify Services
```bash
# Check service status
make ps

# View logs
make logs
```

## Access Services

Once started, access these URLs:

| Service | URL | Credentials |
|---------|-----|-------------|
| **FastAPI Docs** | http://localhost:8000/docs | - |
| **Streamlit Dashboard** | http://localhost:8501 | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Prometheus** | http://localhost:9090 | - |

## Common Commands

### Service Management
```bash
make up          # Start services
make down        # Stop services
make restart     # Restart services
make ps          # Service status
make logs        # View all logs
make logs-app    # View app logs only
```

### Database Operations
```bash
make shell-db    # PostgreSQL shell
make init-db     # Initialize database
make backup-db   # Create backup
make restore-db FILE=backup.sql  # Restore backup
```

### Development
```bash
make shell-app   # Access app container
make test        # Run tests
make test-cov    # Run tests with coverage
```

### Maintenance
```bash
make clean       # Stop and remove everything
make health      # Check service health
```

## Troubleshooting

### Services won't start
```bash
# Check logs for errors
make logs

# Verify Docker is running
docker ps

# Check available resources
docker system df
```

### Port conflicts
If ports are already in use, edit `.env`:
```bash
API_PORT=8001
DASHBOARD_PORT=8502
```

### Permission errors (Grafana)
```bash
sudo chown -R 472:472 ./data/grafana
```

### Database connection issues
```bash
# Check database logs
make logs-db

# Test connection
docker-compose exec app python -c "from src.config import config; print(config.database_url)"
```

## Next Steps

1. **Configure Grafana Datasources**
   - Login to Grafana at http://localhost:3000
   - Verify TimescaleDB datasource connection
   - Import sample dashboards

2. **Initialize Database Schema**
   ```bash
   make init-db
   ```

3. **Test the API**
   - Visit http://localhost:8000/docs
   - Try the example endpoints

4. **Explore the Dashboard**
   - Visit http://localhost:8501
   - Submit test problems
   - View routing decisions

## Production Deployment

For production, create `docker-compose.prod.yml` with:
- Secure passwords
- Resource limits
- SSL/TLS configuration
- Backup automation
- Log aggregation

Deploy with:
```bash
make prod-up
```

## Support

For issues and questions:
- Check logs: `make logs`
- Review documentation in `docs/`
- Open GitHub issue
