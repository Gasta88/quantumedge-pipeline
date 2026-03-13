"""
Profile Loader for QuantumEdge Pipeline.

Loads company-specific profile YAML files and exposes them as typed Pydantic
models. Supports profile selection via CLI argument (--profile) or environment
variable (QUANTUMEDGE_PROFILE), with fallback to 'default'.

Usage:
    >>> from src.profile_loader import load_profile, get_active_profile
    >>> profile = load_profile("rotonium")
    >>> print(profile.name)       # "Rotonium"
    >>> print(profile.tagline)    # "Edge Quantum Computing - OAM Photonic QPU"

    # Or use the auto-detected active profile
    >>> profile = get_active_profile()
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"


# =============================================================================
# Pydantic Models for Profile Schema
# =============================================================================


class EnergyModelConfig(BaseModel):
    """Energy model configuration for a company profile."""

    framing: str = Field(..., description="Energy framing approach: 'SWaP' or 'PUE'")
    label: str = Field(..., description="Display label for energy metric")
    warn_threshold_pct: int = Field(..., ge=0, le=100, description="Warning threshold percentage")


class RoutingConfig(BaseModel):
    """Routing configuration defaults for a company profile."""

    strategy_default: str = Field(..., description="Default routing strategy")
    key_metric: str = Field(..., description="Primary metric for routing decisions")
    power_unit: str = Field(..., description="Unit for power measurements")


class DeploymentConfig(BaseModel):
    """Deployment profile configuration."""

    primary: str = Field(..., description="Default deployment profile name")
    available: List[str] = Field(..., description="All available deployment profiles")

    @field_validator("available")
    @classmethod
    def validate_primary_in_available(cls, v: List[str], info) -> List[str]:
        """Ensure primary profile is in available list."""
        if "primary" in info.data and info.data["primary"] not in v:
            raise ValueError(
                f"Primary profile '{info.data['primary']}' must be in available list: {v}"
            )
        return v


class DemoScenarioRef(BaseModel):
    """Reference to a demo scenario file."""

    name: str = Field(..., description="Display name for the scenario")
    file: str = Field(..., description="Relative path to scenario JSON file")


class DocsConfig(BaseModel):
    """Documentation configuration for a company profile."""

    integration: str = Field(..., description="Path to integration documentation")
    pitch_focus: str = Field(..., description="Key pitch focus areas")


class CompanyProfile(BaseModel):
    """
    Complete company profile configuration.

    Loaded from YAML files in the profiles/ directory. Each profile defines
    company-specific branding, deployment targets, routing defaults, energy
    models, demo scenarios, and documentation references.
    """

    name: str = Field(..., description="Company display name")
    tagline: str = Field(..., description="One-line company description")
    hardware_backend: str = Field(..., description="Backend identifier for hardware layer")
    deployment_profiles: DeploymentConfig
    routing: RoutingConfig
    energy_model: EnergyModelConfig
    demo_scenarios: List[DemoScenarioRef]
    docs: DocsConfig

    class Config:
        extra = "ignore"


# =============================================================================
# Profile Loading Functions
# =============================================================================


def load_profile(profile_name: str) -> CompanyProfile:
    """
    Load and validate a company profile from YAML.

    Args:
        profile_name: Name of the profile (without .yaml extension).
                      Looks for profiles/<profile_name>.yaml

    Returns:
        Validated CompanyProfile instance.

    Raises:
        FileNotFoundError: If the profile YAML file does not exist.
        ValueError: If the YAML content fails Pydantic validation.
    """
    profile_path = PROFILES_DIR / f"{profile_name}.yaml"

    if not profile_path.exists():
        available = [p.stem for p in PROFILES_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Profile '{profile_name}' not found at {profile_path}. Available profiles: {available}"
        )

    logger.info(f"Loading profile '{profile_name}' from {profile_path}")

    with open(profile_path, "r") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(
            f"Profile '{profile_name}' YAML must be a mapping, got {type(raw_data).__name__}"
        )

    try:
        profile = CompanyProfile(**raw_data)
    except Exception as e:
        raise ValueError(f"Profile '{profile_name}' validation failed: {e}") from e

    logger.info(f"Loaded profile: {profile.name} - {profile.tagline}")
    return profile


def resolve_profile_name() -> str:
    """
    Determine which profile to load based on CLI args and env var.

    Resolution order (first match wins):
        1. --profile <name> CLI argument (parsed from sys.argv)
        2. QUANTUMEDGE_PROFILE environment variable
        3. Falls back to 'default'

    Returns:
        Profile name string.
    """
    # 1. Check CLI arguments for --profile
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--profile" and i + 1 < len(args):
            profile_name = args[i + 1]
            logger.info(f"Profile selected via --profile flag: {profile_name}")
            return profile_name

    # 2. Check environment variable
    env_profile = os.environ.get("QUANTUMEDGE_PROFILE", "").strip()
    if env_profile:
        logger.info(f"Profile selected via QUANTUMEDGE_PROFILE env var: {env_profile}")
        return env_profile

    # 3. Default
    logger.info("No profile specified, using 'default'")
    return "default"


# =============================================================================
# Singleton Active Profile
# =============================================================================

_active_profile: Optional[CompanyProfile] = None


def get_active_profile() -> CompanyProfile:
    """
    Get the currently active company profile (singleton).

    Lazily loads the profile on first access using resolve_profile_name().
    Subsequent calls return the cached instance.

    Returns:
        The active CompanyProfile instance.
    """
    global _active_profile
    if _active_profile is None:
        profile_name = resolve_profile_name()
        _active_profile = load_profile(profile_name)
    return _active_profile


def set_active_profile(profile_name: str) -> CompanyProfile:
    """
    Explicitly set the active profile (useful for testing or runtime switching).

    Args:
        profile_name: Name of the profile to load and activate.

    Returns:
        The newly activated CompanyProfile instance.
    """
    global _active_profile
    _active_profile = load_profile(profile_name)
    return _active_profile


def reset_active_profile() -> None:
    """Reset the active profile singleton (useful for testing)."""
    global _active_profile
    _active_profile = None
