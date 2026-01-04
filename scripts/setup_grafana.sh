#!/bin/bash

###########################################
# Grafana Setup Script
# 
# This script configures Grafana with:
# - PostgreSQL datasource
# - QuantumEdge Pipeline Performance dashboard
# - Default dashboard settings
# - Refresh intervals
###########################################

set -e  # Exit on error

# Configuration
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-quantumedge}"
POSTGRES_USER="${POSTGRES_USER:-quantumedge}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-quantumedge123}"
DASHBOARD_JSON="${DASHBOARD_JSON:-dashboard/grafana_dashboards.json}"
MAX_RETRIES=30
RETRY_INTERVAL=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Wait for Grafana to be ready
wait_for_grafana() {
    log_info "Waiting for Grafana to be ready at ${GRAFANA_URL}..."
    
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -s -o /dev/null -w "%{http_code}" "${GRAFANA_URL}/api/health" | grep -q "200"; then
            log_info "Grafana is ready!"
            return 0
        fi
        
        retries=$((retries + 1))
        log_warning "Grafana not ready yet. Attempt ${retries}/${MAX_RETRIES}. Retrying in ${RETRY_INTERVAL}s..."
        sleep $RETRY_INTERVAL
    done
    
    log_error "Grafana failed to become ready after ${MAX_RETRIES} attempts"
    return 1
}

# Configure PostgreSQL datasource
configure_datasource() {
    log_info "Configuring PostgreSQL datasource..."
    
    # Check if datasource already exists
    local datasource_check=$(curl -s -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        "${GRAFANA_URL}/api/datasources/name/PostgreSQL" 2>/dev/null || echo "")
    
    if echo "$datasource_check" | grep -q '"id"'; then
        log_warning "PostgreSQL datasource already exists. Updating..."
        local datasource_id=$(echo "$datasource_check" | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
        
        # Update existing datasource
        local response=$(curl -s -w "\n%{http_code}" -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
            -X PUT \
            -H "Content-Type: application/json" \
            -d '{
                "id": '"$datasource_id"',
                "name": "PostgreSQL",
                "type": "postgres",
                "uid": "postgres-datasource",
                "access": "proxy",
                "url": "'"${POSTGRES_HOST}:${POSTGRES_PORT}"'",
                "database": "'"${POSTGRES_DB}"'",
                "user": "'"${POSTGRES_USER}"'",
                "secureJsonData": {
                    "password": "'"${POSTGRES_PASSWORD}"'"
                },
                "jsonData": {
                    "sslmode": "disable",
                    "maxOpenConns": 0,
                    "maxIdleConns": 2,
                    "connMaxLifetime": 14400,
                    "postgresVersion": 1300,
                    "timescaledb": false
                },
                "isDefault": true
            }' \
            "${GRAFANA_URL}/api/datasources/${datasource_id}")
    else
        # Create new datasource
        local response=$(curl -s -w "\n%{http_code}" -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d '{
                "name": "PostgreSQL",
                "type": "postgres",
                "uid": "postgres-datasource",
                "access": "proxy",
                "url": "'"${POSTGRES_HOST}:${POSTGRES_PORT}"'",
                "database": "'"${POSTGRES_DB}"'",
                "user": "'"${POSTGRES_USER}"'",
                "secureJsonData": {
                    "password": "'"${POSTGRES_PASSWORD}"'"
                },
                "jsonData": {
                    "sslmode": "disable",
                    "maxOpenConns": 0,
                    "maxIdleConns": 2,
                    "connMaxLifetime": 14400,
                    "postgresVersion": 1300,
                    "timescaledb": false
                },
                "isDefault": true
            }' \
            "${GRAFANA_URL}/api/datasources")
    fi
    
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')
    
    if [[ "$http_code" =~ ^2[0-9]{2}$ ]]; then
        log_info "PostgreSQL datasource configured successfully"
        return 0
    else
        log_error "Failed to configure datasource. HTTP code: ${http_code}"
        log_error "Response: ${body}"
        return 1
    fi
}

# Import dashboard
import_dashboard() {
    log_info "Importing QuantumEdge Pipeline Performance dashboard..."
    
    if [ ! -f "$DASHBOARD_JSON" ]; then
        log_error "Dashboard JSON file not found: ${DASHBOARD_JSON}"
        return 1
    fi
    
    # Read dashboard JSON
    local dashboard_content=$(cat "$DASHBOARD_JSON")
    
    # Import dashboard
    local response=$(curl -s -w "\n%{http_code}" -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$dashboard_content" \
        "${GRAFANA_URL}/api/dashboards/db")
    
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')
    
    if [[ "$http_code" =~ ^2[0-9]{2}$ ]]; then
        log_info "Dashboard imported successfully"
        
        # Extract dashboard UID for setting as default
        local dashboard_uid=$(echo "$body" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
        if [ -n "$dashboard_uid" ]; then
            log_info "Dashboard UID: ${dashboard_uid}"
            echo "$dashboard_uid" > /tmp/grafana_dashboard_uid.txt
        fi
        return 0
    else
        log_error "Failed to import dashboard. HTTP code: ${http_code}"
        log_error "Response: ${body}"
        return 1
    fi
}

# Set default dashboard
set_default_dashboard() {
    log_info "Setting default dashboard and preferences..."
    
    # Get organization ID
    local org_response=$(curl -s -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        "${GRAFANA_URL}/api/user/orgs")
    
    local org_id=$(echo "$org_response" | grep -o '"orgId":[0-9]*' | head -1 | grep -o '[0-9]*')
    
    if [ -z "$org_id" ]; then
        log_warning "Could not determine organization ID. Using default (1)"
        org_id=1
    fi
    
    # Update organization preferences
    local response=$(curl -s -w "\n%{http_code}" -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        -X PUT \
        -H "Content-Type: application/json" \
        -d '{
            "theme": "",
            "homeDashboardId": 0,
            "timezone": "browser"
        }' \
        "${GRAFANA_URL}/api/org/preferences")
    
    local http_code=$(echo "$response" | tail -n1)
    
    if [[ "$http_code" =~ ^2[0-9]{2}$ ]]; then
        log_info "Organization preferences updated successfully"
    else
        log_warning "Failed to update organization preferences"
    fi
    
    # Update user preferences for auto-refresh
    local user_response=$(curl -s -w "\n%{http_code}" -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        -X PUT \
        -H "Content-Type: application/json" \
        -d '{
            "theme": "",
            "homeDashboardId": 0,
            "timezone": "browser"
        }' \
        "${GRAFANA_URL}/api/user/preferences")
    
    local user_http_code=$(echo "$user_response" | tail -n1)
    
    if [[ "$user_http_code" =~ ^2[0-9]{2}$ ]]; then
        log_info "User preferences updated successfully"
    else
        log_warning "Failed to update user preferences"
    fi
}

# Configure refresh intervals
configure_refresh_intervals() {
    log_info "Configuring dashboard refresh intervals..."
    
    # Update Grafana configuration for allowed refresh intervals
    log_info "Refresh intervals should be configured in grafana.ini:"
    log_info "  [dashboards]"
    log_info "  min_refresh_interval = 5s"
    log_info ""
    log_info "Default dashboard refresh is set to 30s in the dashboard JSON"
}

# Main execution
main() {
    log_info "Starting Grafana setup..."
    log_info "================================================"
    
    # Step 1: Wait for Grafana
    if ! wait_for_grafana; then
        log_error "Setup failed: Grafana is not available"
        exit 1
    fi
    
    # Step 2: Configure datasource
    if ! configure_datasource; then
        log_error "Setup failed: Could not configure datasource"
        exit 1
    fi
    
    # Step 3: Import dashboard
    if ! import_dashboard; then
        log_error "Setup failed: Could not import dashboard"
        exit 1
    fi
    
    # Step 4: Set default dashboard
    set_default_dashboard
    
    # Step 5: Configure refresh intervals
    configure_refresh_intervals
    
    log_info "================================================"
    log_info "Grafana setup completed successfully!"
    log_info ""
    log_info "Access Grafana at: ${GRAFANA_URL}"
    log_info "Username: ${GRAFANA_USER}"
    log_info "Password: ${GRAFANA_PASSWORD}"
    log_info ""
    log_info "Dashboard: QuantumEdge Pipeline Performance"
    log_info "Datasource: PostgreSQL (${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB})"
}

# Run main function
main "$@"
