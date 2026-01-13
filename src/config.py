"""
Configuration Management for QuantumEdge Pipeline.

This module provides a type-safe, validated configuration system using Pydantic.
Configuration values are loaded from environment variables or .env file with
intelligent defaults for development.

Usage:
    >>> from src.config import settings
    >>> print(settings.database.url)
    >>> print(settings.quantum.shots)
    >>> print(settings.edge_profiles["aerospace"].power_budget_watts)
"""

from typing import Dict, List, Optional, Literal
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Database Configuration
# =============================================================================

class DatabaseConfig(BaseSettings):
    """
    Database connection configuration for TimescaleDB/PostgreSQL.
    
    Manages connection parameters, pooling settings, and URL construction
    for both sync and async database operations.
    
    Environment Variables:
        POSTGRES_HOST: Database server hostname (default: localhost)
        POSTGRES_PORT: Database server port (default: 5432)
        POSTGRES_DB: Database name (default: quantumedge)
        POSTGRES_USER: Authentication username (default: qe_user)
        POSTGRES_PASSWORD: Authentication password (default: qe_pass)
    
    Example:
        >>> db_config = DatabaseConfig()
        >>> print(db_config.url)  # postgresql://qe_user:qe_pass@localhost:5432/quantumedge
        >>> print(db_config.async_url)  # postgresql+asyncpg://...
    """
    
    # Database server hostname or IP address
    host: str = Field(
        default="localhost",
        description="PostgreSQL server hostname",
        env="POSTGRES_HOST"
    )
    
    # Database server port
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL server port",
        env="POSTGRES_PORT"
    )
    
    # Database name
    database: str = Field(
        default="quantumedge",
        description="Database name for the application",
        env="POSTGRES_DB"
    )
    
    # Authentication username
    user: str = Field(
        default="qe_user",
        description="Database authentication username",
        env="POSTGRES_USER"
    )
    
    # Authentication password (should be changed in production!)
    password: str = Field(
        default="qe_pass",
        description="Database authentication password",
        env="POSTGRES_PASSWORD"
    )
    
    # Connection pool size for async operations
    pool_size: int = Field(
        default=10,
        ge=1,
        description="Number of connections to maintain in pool"
    )
    
    # Maximum overflow connections for burst traffic
    max_overflow: int = Field(
        default=20,
        ge=0,
        description="Additional connections beyond pool_size"
    )
    
    # Connection timeout in seconds
    pool_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Seconds to wait for available connection"
    )
    
    # Connection recycle time in seconds (prevents stale connections)
    pool_recycle: int = Field(
        default=3600,
        ge=300,
        description="Seconds before recycling connections"
    )
    
    @computed_field
    @property
    def url(self) -> str:
        """Construct synchronous database URL for psycopg2."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @computed_field
    @property
    def async_url(self) -> str:
        """Construct async database URL for asyncpg."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# =============================================================================
# Quantum Computing Configuration
# =============================================================================

class QuantumConfig(BaseSettings):
    """
    Quantum computing and simulation configuration.
    
    Controls quantum backend selection, circuit execution parameters,
    noise modeling, and simulator settings for quantum optimization.
    
    Environment Variables:
        QUANTUM_BACKEND: Backend type (simulator, ibm_quantum, aws_braket)
        QUANTUM_SHOTS: Number of circuit executions (default: 1000)
        QUANTUM_NOISE_MODEL: Noise simulation (none, depolarizing, thermal)
    
    Example:
        >>> quantum_config = QuantumConfig()
        >>> print(quantum_config.backend)  # simulator
        >>> print(quantum_config.shots)  # 1000
        >>> print(quantum_config.is_noisy)  # False
    """
    
    # Quantum backend selection
    backend: Literal["simulator", "ibm_quantum", "aws_braket"] = Field(
        default="simulator",
        description="Quantum backend: simulator (local), ibm_quantum (real hardware), aws_braket (AWS)",
        env="QUANTUM_BACKEND"
    )
    
    # Number of circuit executions for measurement statistics
    shots: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Number of shots (circuit executions) for quantum measurements",
        env="QUANTUM_SHOTS"
    )
    
    # Noise model for realistic quantum simulation
    noise_model: Literal["none", "depolarizing", "thermal"] = Field(
        default="none",
        description="Noise model: none (ideal), depolarizing (gate errors), thermal (T1/T2 decay)",
        env="QUANTUM_NOISE_MODEL"
    )
    
    # Maximum number of qubits supported
    max_qubits: int = Field(
        default=20,
        ge=2,
        le=50,
        description="Maximum qubits for quantum circuits (limited by simulator)",
        env="MAX_QUBITS"
    )
    
    # Optimization level for circuit transpilation (0-3)
    optimization_level: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Circuit optimization level: 0 (none) to 3 (aggressive)",
        env="QUANTUM_OPTIMIZATION_LEVEL"
    )
    
    # Depolarizing noise error rate (if noise_model = depolarizing)
    depolarizing_error_rate: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Gate error rate for depolarizing noise (0.001 = 0.1%)",
        env="QUANTUM_DEPOLARIZING_ERROR"
    )
    
    # T1 relaxation time in microseconds (if noise_model = thermal)
    t1_relaxation_us: float = Field(
        default=50.0,
        ge=1.0,
        description="T1 amplitude damping time in microseconds",
        env="QUANTUM_T1_RELAXATION"
    )
    
    # T2 dephasing time in microseconds (if noise_model = thermal)
    t2_dephasing_us: float = Field(
        default=70.0,
        ge=1.0,
        description="T2 phase damping time in microseconds",
        env="QUANTUM_T2_DEPHASING"
    )
    
    @computed_field
    @property
    def is_noisy(self) -> bool:
        """Check if noise simulation is enabled."""
        return self.noise_model != "none"
    
    @computed_field
    @property
    def is_real_hardware(self) -> bool:
        """Check if using real quantum hardware."""
        return self.backend in ["ibm_quantum", "aws_braket"]
    
    @field_validator("t2_dephasing_us")
    @classmethod
    def validate_t2_le_t1(cls, v: float, info) -> float:
        """Ensure T2 <= 2*T1 (physical constraint)."""
        if "t1_relaxation_us" in info.data:
            t1 = info.data["t1_relaxation_us"]
            if v > 2 * t1:
                raise ValueError(f"T2 ({v}) must be <= 2*T1 ({2*t1})")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# =============================================================================
# Edge Computing Profile Configuration
# =============================================================================

class EdgeProfile(BaseSettings):
    """
    Edge computing environment resource constraints.
    
    Defines power budget, thermal limits, memory, CPU, and timeout constraints
    for different edge deployment scenarios (aerospace, mobile, ground_server).
    
    Attributes:
        power_budget_watts: Maximum power consumption in watts
        thermal_limit_celsius: Temperature threshold before throttling
        memory_mb: Available RAM in megabytes
        cpu_cores: Number of CPU cores available
        max_execution_time_sec: Maximum job execution time in seconds
        network_latency_ms: Expected network round-trip time
    """
    
    power_budget_watts: float = Field(
        ge=1.0,
        description="Power budget in watts"
    )
    
    thermal_limit_celsius: float = Field(
        ge=40.0,
        le=120.0,
        description="Maximum temperature in Celsius"
    )
    
    memory_mb: int = Field(
        ge=512,
        description="Available memory in megabytes"
    )
    
    cpu_cores: int = Field(
        ge=1,
        description="Number of CPU cores"
    )
    
    max_execution_time_sec: int = Field(
        ge=1,
        description="Maximum execution time in seconds"
    )
    
    network_latency_ms: int = Field(
        ge=1,
        description="Network latency in milliseconds"
    )
    
    model_config = SettingsConfigDict(extra="ignore")


class EdgeConfig(BaseSettings):
    """
    Edge computing profiles for different deployment scenarios.
    
    Provides predefined resource constraint profiles optimized for:
    - Aerospace: Strict power limits, moderate compute
    - Mobile: Battery constrained, low power
    - Ground: Relaxed constraints, high compute
    
    Environment Variables:
        EDGE_POWER_BUDGET_WATTS: Override default power budget
        EDGE_THERMAL_LIMIT_CELSIUS: Override default thermal limit
        EDGE_MEMORY_MB: Override default memory
        EDGE_CPU_CORES: Override default CPU cores
        EDGE_MAX_EXECUTION_TIME_SEC: Override default timeout
    
    Example:
        >>> edge_config = EdgeConfig()
        >>> aerospace = edge_config.profiles["aerospace"]
        >>> print(f"Aerospace power budget: {aerospace.power_budget_watts}W")
        >>> print(f"Aerospace timeout: {aerospace.max_execution_time_sec}s")
    """
    
    # Default edge profile to use if not specified
    default_profile: Literal["aerospace", "mobile", "ground_server"] = Field(
        default="ground_server",
        description="Default edge profile to use",
        env="EDGE_DEFAULT_PROFILE"
    )
    
    # Custom overrides for power budget (overrides profile defaults)
    power_budget_watts: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="Override power budget in watts",
        env="EDGE_POWER_BUDGET_WATTS"
    )
    
    # Custom overrides for thermal limit
    thermal_limit_celsius: Optional[float] = Field(
        default=None,
        ge=40.0,
        le=120.0,
        description="Override thermal limit in Celsius",
        env="EDGE_THERMAL_LIMIT_CELSIUS"
    )
    
    # Custom overrides for memory
    memory_mb: Optional[int] = Field(
        default=None,
        ge=512,
        description="Override memory in megabytes",
        env="EDGE_MEMORY_MB"
    )
    
    # Custom overrides for CPU cores
    cpu_cores: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override CPU cores",
        env="EDGE_CPU_CORES"
    )
    
    # Custom overrides for execution timeout
    max_execution_time_sec: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override maximum execution time in seconds",
        env="EDGE_MAX_EXECUTION_TIME_SEC"
    )
    
    @computed_field
    @property
    def profiles(self) -> Dict[str, EdgeProfile]:
        """
        Get predefined edge computing profiles.
        
        Profiles:
            aerospace: Satellite/aircraft - strict power (50W), moderate compute, 10s timeout
            mobile: Smartphone/tablet - battery limited (15W), low compute, 5s timeout
            ground_server: Data center/server - relaxed (200W), high compute, 60s timeout
        
        Returns:
            Dictionary mapping profile names to EdgeProfile configurations
        """
        return {
            "aerospace": EdgeProfile(
                power_budget_watts=self.power_budget_watts or 50.0,
                thermal_limit_celsius=self.thermal_limit_celsius or 70.0,
                memory_mb=self.memory_mb or 2048,
                cpu_cores=self.cpu_cores or 2,
                max_execution_time_sec=self.max_execution_time_sec or 10,
                network_latency_ms=500,  # High latency (satellite)
            ),
            "mobile": EdgeProfile(
                power_budget_watts=self.power_budget_watts or 15.0,
                thermal_limit_celsius=self.thermal_limit_celsius or 45.0,
                memory_mb=self.memory_mb or 1024,
                cpu_cores=self.cpu_cores or 4,
                max_execution_time_sec=self.max_execution_time_sec or 5,
                network_latency_ms=100,  # Cellular latency
            ),
            "ground_server": EdgeProfile(
                power_budget_watts=self.power_budget_watts or 200.0,
                thermal_limit_celsius=self.thermal_limit_celsius or 85.0,
                memory_mb=self.memory_mb or 8192,
                cpu_cores=self.cpu_cores or 8,
                max_execution_time_sec=self.max_execution_time_sec or 60,
                network_latency_ms=10,  # Low latency (LAN)
            ),
        }
    
    def get_profile(self, profile_name: str) -> EdgeProfile:
        """
        Get a specific edge profile by name.
        
        Args:
            profile_name: Profile name ('aerospace', 'mobile', 'ground_server')
        
        Returns:
            EdgeProfile configuration
        
        Raises:
            KeyError: If profile name is invalid
        """
        if profile_name not in self.profiles:
            raise KeyError(f"Invalid profile '{profile_name}'. Must be one of {list(self.profiles.keys())}")
        return self.profiles[profile_name]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# =============================================================================
# Router Configuration
# =============================================================================

class RouterConfig(BaseSettings):
    """
    Quantum-classical routing decision configuration.
    
    Defines thresholds, rules, and heuristics for deciding whether to
    route optimization problems to classical, quantum, or hybrid solvers.
    
    Environment Variables:
        ROUTER_CLASSICAL_THRESHOLD: Max problem size for classical (default: 100)
        ROUTER_QUANTUM_THRESHOLD: Max problem size for quantum (default: 1000)
        ROUTER_HYBRID_ENABLED: Enable hybrid routing (default: true)
    
    Example:
        >>> router_config = RouterConfig()
        >>> if problem_size < router_config.classical_threshold:
        ...     solver = "classical"
        >>> elif problem_size < router_config.quantum_threshold:
        ...     solver = "quantum"
    """
    
    # Maximum problem size to route to classical solver only
    classical_threshold: int = Field(
        default=100,
        ge=10,
        description="Max problem size for pure classical routing (nodes/variables)",
        env="ROUTER_CLASSICAL_THRESHOLD"
    )
    
    # Maximum problem size feasible for quantum solver
    quantum_threshold: int = Field(
        default=1000,
        ge=10,
        description="Max problem size for quantum routing (limited by qubit count)",
        env="ROUTER_QUANTUM_THRESHOLD"
    )
    
    # Enable hybrid quantum-classical routing
    hybrid_enabled: bool = Field(
        default=True,
        description="Enable hybrid solver that combines classical and quantum",
        env="ROUTER_HYBRID_ENABLED"
    )
    
    # Minimum quantum advantage ratio to prefer quantum (speedup multiplier)
    min_quantum_advantage: float = Field(
        default=1.2,
        ge=1.0,
        description="Minimum speedup ratio to justify quantum routing (e.g., 1.2 = 20% faster)",
        env="ROUTER_MIN_QUANTUM_ADVANTAGE"
    )
    
    # Maximum energy budget ratio (quantum_energy / classical_energy)
    max_energy_ratio: float = Field(
        default=2.0,
        ge=0.5,
        description="Max energy consumption ratio quantum/classical (2.0 = allow 2x energy)",
        env="ROUTER_MAX_ENERGY_RATIO"
    )
    
    # Prefer quantum for sparse problems (sparsity threshold)
    sparse_quantum_preference: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sparsity threshold for quantum preference (0.3 = <30% density)",
        env="ROUTER_SPARSE_PREFERENCE"
    )
    
    @field_validator("quantum_threshold")
    @classmethod
    def validate_quantum_ge_classical(cls, v: int, info) -> int:
        """Ensure quantum_threshold >= classical_threshold."""
        if "classical_threshold" in info.data:
            classical = info.data["classical_threshold"]
            if v < classical:
                raise ValueError(
                    f"quantum_threshold ({v}) must be >= classical_threshold ({classical})"
                )
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# =============================================================================
# API Configuration
# =============================================================================

class APIConfig(BaseSettings):
    """
    FastAPI application configuration.
    
    Controls API server settings, CORS, security, logging, and
    performance parameters for the REST API.
    
    Environment Variables:
        API_HOST: Server bind address (default: 0.0.0.0)
        API_PORT: Server port (default: 8000)
        DEBUG: Enable debug mode (default: true)
        LOG_LEVEL: Logging level (default: INFO)
    
    Example:
        >>> api_config = APIConfig()
        >>> app = FastAPI(
        ...     title=api_config.title,
        ...     debug=api_config.debug,
        ...     docs_url=api_config.docs_url
        ... )
    """
    
    # API server bind address (0.0.0.0 = all interfaces, 127.0.0.1 = localhost only)
    host: str = Field(
        default="0.0.0.0",
        description="API server bind address",
        env="API_HOST"
    )
    
    # API server port
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port",
        env="API_PORT"
    )
    
    # Number of uvicorn worker processes
    workers: int = Field(
        default=4,
        ge=1,
        description="Number of uvicorn worker processes",
        env="API_WORKERS"
    )
    
    # Application name
    title: str = Field(
        default="QuantumEdge Pipeline API",
        description="API title displayed in docs"
    )
    
    # Application version
    version: str = Field(
        default="0.1.0",
        description="API version"
    )
    
    # Enable debug mode (detailed errors, auto-reload)
    debug: bool = Field(
        default=True,
        description="Enable debug mode with detailed errors",
        env="DEBUG"
    )
    
    # Logging level
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
        env="LOG_LEVEL"
    )
    
    # OpenAPI documentation URL
    docs_url: str = Field(
        default="/docs",
        description="Swagger UI documentation endpoint"
    )
    
    # ReDoc documentation URL
    redoc_url: str = Field(
        default="/redoc",
        description="ReDoc documentation endpoint"
    )
    
    # CORS allowed origins
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="CORS allowed origins for frontend access",
        env="CORS_ORIGINS"
    )
    
    # CORS allow credentials
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow cookies in CORS requests"
    )
    
    # CORS allowed methods
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods for CORS"
    )
    
    # CORS allowed headers
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed headers for CORS"
    )
    
    # Secret key for JWT/sessions (MUST change in production!)
    secret_key: str = Field(
        default="change-me-in-production-use-strong-random-key",
        min_length=32,
        description="Secret key for cryptographic operations",
        env="SECRET_KEY"
    )
    
    # Request rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable API rate limiting"
    )
    
    # Maximum requests per minute per IP
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Max requests per minute per IP address"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# =============================================================================
# Global Settings Container
# =============================================================================

class Settings(BaseSettings):
    """
    Global application settings container.
    
    Aggregates all configuration sections into a single settings object
    for convenient access throughout the application.
    
    Usage:
        >>> from src.config import settings
        >>> 
        >>> # Access database configuration
        >>> db_url = settings.database.async_url
        >>> 
        >>> # Access quantum configuration
        >>> shots = settings.quantum.shots
        >>> 
        >>> # Access edge profiles
        >>> aerospace_profile = settings.edge.profiles["aerospace"]
        >>> power_limit = aerospace_profile.power_budget_watts
        >>> 
        >>> # Access router configuration
        >>> classical_threshold = settings.router.classical_threshold
        >>> 
        >>> # Access API configuration
        >>> api_host = settings.api.host
    """
    
    # Database configuration
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Quantum computing configuration
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    
    # Edge computing profiles
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    
    # Routing configuration
    router: RouterConfig = Field(default_factory=RouterConfig)
    
    # API configuration
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Application environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
        env="APP_ENV"
    )
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Singleton settings instance - import this throughout the application
settings = Settings()


# Convenience exports for backward compatibility
config = settings  # Alias for legacy code
