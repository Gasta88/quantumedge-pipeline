"""
Unit tests for profile loading and validation.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.profile_loader import (
    load_profile,
    resolve_profile_name,
    get_active_profile,
    set_active_profile,
    reset_active_profile,
    CompanyProfile,
    PROFILES_DIR,
)


class TestLoadProfile:
    """Tests for load_profile()."""

    def test_load_edge_profile(self):
        """Load edge profile and verify all fields."""
        profile = load_profile("edge")

        assert profile.name == "Edge Photonics"
        assert profile.tagline == "Edge Quantum Computing - Photonic QPU"
        assert profile.hardware_backend == "photonic_mock"
        assert profile.deployment_profiles.primary == "aerospace"
        assert "aerospace" in profile.deployment_profiles.available
        assert "mobile" in profile.deployment_profiles.available
        assert "ground_server" in profile.deployment_profiles.available
        assert profile.routing.strategy_default == "energy_optimized"
        assert profile.routing.key_metric == "swap_score"
        assert profile.energy_model.framing == "SWaP"
        assert profile.energy_model.label == "Battery Budget Used"
        assert profile.energy_model.warn_threshold_pct == 80
        assert len(profile.demo_scenarios) == 3
        assert profile.docs.integration == "docs/edge-integration.md"

    def test_load_datacenter_profile(self):
        """Load datacenter profile and verify all fields."""
        profile = load_profile("datacenter")

        assert profile.name == "Data Center Photonics"
        assert profile.hardware_backend == "photonic_cloud"
        assert profile.deployment_profiles.primary == "hpc_cluster"
        assert "hpc_cluster" in profile.deployment_profiles.available
        assert "datacenter_rack" in profile.deployment_profiles.available
        assert "cloud_node" in profile.deployment_profiles.available
        assert profile.routing.strategy_default == "cost_per_job"
        assert profile.routing.key_metric == "throughput"
        assert profile.routing.power_unit == "pue_adjusted"
        assert profile.energy_model.framing == "PUE"
        assert profile.energy_model.warn_threshold_pct == 90
        assert len(profile.demo_scenarios) == 3
        assert profile.docs.integration == "docs/datacenter-integration.md"

    def test_load_default_profile(self):
        """Default profile should match Edge Photonics behaviour."""
        default = load_profile("default")
        edge = load_profile("edge")

        assert default.name == "QuantumEdge"
        assert default.hardware_backend == edge.hardware_backend
        assert default.deployment_profiles.primary == edge.deployment_profiles.primary
        assert default.energy_model.framing == edge.energy_model.framing

    def test_invalid_profile_name(self):
        """Non-existent profile should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_profile("nonexistent_profile_xyz")

    def test_invalid_yaml_content(self, tmp_path):
        """Malformed YAML should raise ValueError."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("- this\n- is\n- a list\n")

        with patch("src.profile_loader.PROFILES_DIR", tmp_path):
            with pytest.raises(ValueError, match="must be a mapping"):
                load_profile("bad")

    def test_missing_required_fields(self, tmp_path):
        """YAML missing required fields should raise ValueError."""
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("name: TestOnly\n")

        with patch("src.profile_loader.PROFILES_DIR", tmp_path):
            with pytest.raises(ValueError, match="validation failed"):
                load_profile("incomplete")

    def test_profile_returns_company_profile_type(self):
        """Returned object should be a CompanyProfile instance."""
        profile = load_profile("edge")
        assert isinstance(profile, CompanyProfile)

    def test_demo_scenarios_have_name_and_file(self):
        """Every demo scenario must have a name and file path."""
        for profile_name in ("edge", "datacenter"):
            profile = load_profile(profile_name)
            for scenario in profile.demo_scenarios:
                assert scenario.name, f"Scenario in {profile_name} missing name"
                assert scenario.file, f"Scenario in {profile_name} missing file"

    def test_extra_fields_ignored(self, tmp_path):
        """Extra unknown fields in YAML should be silently ignored."""
        extra = tmp_path / "extra.yaml"
        extra.write_text(
            "name: Test\n"
            "tagline: T\n"
            "hardware_backend: test\n"
            "deployment_profiles:\n"
            "  primary: a\n"
            "  available: [a]\n"
            "routing:\n"
            "  strategy_default: x\n"
            "  key_metric: y\n"
            "  power_unit: z\n"
            "energy_model:\n"
            "  framing: F\n"
            "  label: L\n"
            "  warn_threshold_pct: 50\n"
            "demo_scenarios: []\n"
            "docs:\n"
            "  integration: d.md\n"
            "  pitch_focus: p\n"
            "unknown_field: should_be_ignored\n"
        )
        with patch("src.profile_loader.PROFILES_DIR", tmp_path):
            profile = load_profile("extra")
            assert profile.name == "Test"


class TestResolveProfileName:
    """Tests for resolve_profile_name()."""

    def test_env_var_override(self):
        """QUANTUMEDGE_PROFILE env var should be respected."""
        with patch.dict(os.environ, {"QUANTUMEDGE_PROFILE": "datacenter"}):
            with patch("sys.argv", ["app"]):
                assert resolve_profile_name() == "datacenter"

    def test_cli_arg_override(self):
        """--profile CLI arg should take precedence over env var."""
        with patch.dict(os.environ, {"QUANTUMEDGE_PROFILE": "edge"}):
            with patch("sys.argv", ["app", "--profile", "datacenter"]):
                assert resolve_profile_name() == "datacenter"

    def test_default_fallback(self):
        """Without CLI or env var, should return 'default'."""
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("QUANTUMEDGE_PROFILE", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("sys.argv", ["app"]):
                    assert resolve_profile_name() == "default"

    def test_empty_env_var_falls_back(self):
        """Empty QUANTUMEDGE_PROFILE should fall back to default."""
        with patch.dict(os.environ, {"QUANTUMEDGE_PROFILE": ""}):
            with patch("sys.argv", ["app"]):
                assert resolve_profile_name() == "default"


class TestActiveProfile:
    """Tests for singleton active profile."""

    def setup_method(self):
        reset_active_profile()

    def teardown_method(self):
        reset_active_profile()

    def test_set_active_profile(self):
        """set_active_profile should load and cache the profile."""
        profile = set_active_profile("datacenter")
        assert profile.name == "Data Center Photonics"
        # Subsequent get should return same object
        assert get_active_profile() is profile

    def test_get_active_profile_default(self):
        """get_active_profile without set should auto-resolve."""
        with patch("sys.argv", ["app"]):
            with patch.dict(os.environ, {"QUANTUMEDGE_PROFILE": "edge"}):
                profile = get_active_profile()
                assert profile.name == "Edge Photonics"

    def test_reset_clears_cache(self):
        """reset_active_profile should clear the singleton."""
        set_active_profile("datacenter")
        reset_active_profile()
        with patch("sys.argv", ["app"]):
            with patch.dict(os.environ, {"QUANTUMEDGE_PROFILE": "edge"}):
                profile = get_active_profile()
                assert profile.name == "Edge Photonics"
