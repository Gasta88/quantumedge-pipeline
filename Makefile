.PHONY: help build up down restart logs ps clean init-db shell-db shell-redis shell-app test

# Default target
help:
	@echo "QuantumEdge Pipeline - Docker Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs (all services)"
	@echo "  make ps         - Show service status"
	@echo "  make clean      - Stop and remove volumes"
	@echo "  make init-db    - Initialize database"
	@echo "  make shell-db   - Open PostgreSQL shell"
	@echo "  make shell-redis - Open Redis CLI"
	@echo "  make shell-app  - Open app container shell"
	@echo "  make test       - Run tests"
	@echo "  make backup-db  - Backup database"
	@echo "  make restore-db - Restore database from backup"

# Build Docker images
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d
	@echo "Services starting..."
	@echo "Waiting for health checks..."
	@sleep 10
	@docker-compose ps
	@echo ""
	@echo "Services available at:"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Dashboard:  http://localhost:8501"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"

# Stop all services
down:
	docker-compose down

# Restart all services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

# View logs for specific service
logs-app:
	docker-compose logs -f app

logs-db:
	docker-compose logs -f postgres-timescale

logs-grafana:
	docker-compose logs -f grafana

# Show service status
ps:
	docker-compose ps

# Clean everything (including volumes)
clean:
	docker-compose down -v
	@echo "Cleaning local data directories..."
	@rm -rf ./data/postgres/* ./data/grafana/*
	@echo "Clean complete!"

# Initialize database
init-db:
	@echo "Initializing database..."
	docker-compose exec postgres-timescale psql -U qe_user -d quantumedge -f /docker-entrypoint-initdb.d/01-init.sql
	@echo "Database initialized!"

# Database shell
shell-db:
	docker-compose exec postgres-timescale psql -U qe_user -d quantumedge

# Redis CLI
shell-redis:
	docker-compose exec redis redis-cli

# App container shell
shell-app:
	docker-compose exec app bash

# Run tests
test:
	docker-compose exec app pytest tests/ -v

# Run tests with coverage
test-cov:
	docker-compose exec app pytest tests/ -v --cov=src --cov-report=html

# Backup database
backup-db:
	@mkdir -p ./backups
	@echo "Creating database backup..."
	docker-compose exec -T postgres-timescale pg_dump -U qe_user quantumedge > ./backups/quantumedge_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in ./backups/"

# Restore database (specify file with FILE=backup.sql)
restore-db:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify backup file with FILE=path/to/backup.sql"; \
		exit 1; \
	fi
	@echo "Restoring database from $(FILE)..."
	docker-compose exec -T postgres-timescale psql -U qe_user quantumedge < $(FILE)
	@echo "Database restored!"

# View Grafana datasources
grafana-datasources:
	@cat monitoring/grafana/datasources/datasource.yml

# Scale services
scale-app:
	docker-compose up -d --scale app=3

# Health check all services
health:
	@echo "Checking service health..."
	@docker-compose ps | grep "Up"
	@echo ""
	@curl -s http://localhost:8000/health | jq . || echo "App not responding"
	@curl -s http://localhost:9090/-/healthy || echo "Prometheus not responding"

# Install development dependencies
dev-setup:
	@cp .env.example .env
	@echo "Environment file created. Please update .env with your settings."
	@mkdir -p ./data/postgres ./data/grafana ./logs ./backups
	@chmod 777 ./data/grafana
	@echo "Development environment ready!"

# Production build
prod-build:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Production deploy
prod-up:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
