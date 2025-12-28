"""Configuration management for QuantumEdge Pipeline."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    """Application configuration."""

    # Application
    app_name: str = Field(default="quantumedge-pipeline", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    app_env: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=True, env="DEBUG")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")

    # Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="quantumedge", env="POSTGRES_DB")
    postgres_user: str = Field(default="quantumedge_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="changeme", env="POSTGRES_PASSWORD")

    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # Quantum Configuration
    quantum_backend: str = Field(default="simulator", env="QUANTUM_BACKEND")
    quantum_shots: int = Field(default=1000, env="QUANTUM_SHOTS")
    max_qubits: int = Field(default=20, env="MAX_QUBITS")

    # Edge Constraints
    edge_max_memory_mb: int = Field(default=4096, env="EDGE_MAX_MEMORY_MB")
    edge_max_cpu_cores: int = Field(default=4, env="EDGE_MAX_CPU_CORES")
    edge_max_execution_time_sec: int = Field(default=300, env="EDGE_MAX_EXECUTION_TIME_SEC")
    edge_network_latency_ms: int = Field(default=100, env="EDGE_NETWORK_LATENCY_MS")

    # Router Configuration
    router_classical_threshold: int = Field(default=100, env="ROUTER_CLASSICAL_THRESHOLD")
    router_quantum_threshold: int = Field(default=1000, env="ROUTER_QUANTUM_THRESHOLD")
    router_hybrid_enabled: bool = Field(default=True, env="ROUTER_HYBRID_ENABLED")

    # Monitoring
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")

    # Performance
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    async_workers: int = Field(default=10, env="ASYNC_WORKERS")

    # Security
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        """Get PostgreSQL database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Global configuration instance
config = Config()
