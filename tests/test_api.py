"""
Unit tests for API module (minimal coverage with success/failure cases).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import modules to test
from src.api.main import convert_numpy_types, MaxCutJobRequest, health_check
from src.api.orchestrator import JobOrchestrator


class TestConvertNumpyTypes:
    """Test convert_numpy_types utility function."""
    
    def test_convert_numpy_types_success(self):
        """Test successful numpy type conversion."""
        import numpy as np
        
        # Create test data with numpy types
        test_obj = {
            'bool': np.bool_(True),
            'int': np.int64(42),
            'float': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'nested': {
                'value': np.int32(100)
            }
        }
        
        result = convert_numpy_types(test_obj)
        
        # Check conversions
        assert result['bool'] is True
        assert result['int'] == 42
        assert result['float'] == 3.14
        assert result['array'] == [1, 2, 3]
        assert result['nested']['value'] == 100
        
        # Check types are native Python
        assert isinstance(result['bool'], bool)
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['array'], list)
    
    def test_convert_numpy_types_failure(self):
        """Test convert with invalid or unsupported types."""
        # Test with None (should return unchanged)
        result = convert_numpy_types(None)
        assert result is None
        
        # Test with non-numpy object (should return unchanged)
        test_obj = {'normal': 'string', 'number': 42}
        result = convert_numpy_types(test_obj)
        assert result == test_obj


class TestJobOrchestrator:
    """Test JobOrchestrator class."""
    
    def test_orchestrator_initialization_success(self):
        """Test successful orchestrator initialization."""
        orchestrator = JobOrchestrator(
            enable_db=False,
            enable_validation=True,
            default_timeout_s=300.0,
            max_retries=2,
            strategy='balanced'
        )
        
        assert orchestrator.enable_db is False
        assert orchestrator.enable_validation is True
        assert orchestrator.default_timeout_s == 300.0
        assert orchestrator.max_retries == 2
        assert orchestrator.strategy == 'balanced'
        assert orchestrator.analyzer is not None
        assert orchestrator.router is not None
        assert orchestrator.classical_solver is not None
        assert orchestrator.quantum_solver is not None
    
    def test_orchestrator_initialization_failure(self):
        """Test orchestrator initialization with invalid strategy."""
        with pytest.raises(ValueError, match="Invalid routing strategy"):
            JobOrchestrator(
                enable_db=False,
                strategy='invalid_strategy'
            )


class TestMaxCutJobRequest:
    """Test MaxCutJobRequest Pydantic model."""
    
    def test_maxcut_request_validation_success(self):
        """Test successful MaxCut request validation."""
        request = MaxCutJobRequest(
            num_nodes=30,
            edge_probability=0.3,
            edge_profile='aerospace',
            strategy='balanced',
            seed=42
        )
        
        assert request.num_nodes == 30
        assert request.edge_probability == 0.3
        assert request.edge_profile == 'aerospace'
        assert request.strategy == 'balanced'
        assert request.seed == 42
    
    def test_maxcut_request_validation_failure(self):
        """Test MaxCut request validation with invalid values."""
        # Invalid edge_profile
        with pytest.raises(ValueError, match="edge_profile must be one of"):
            MaxCutJobRequest(
                num_nodes=30,
                edge_probability=0.3,
                edge_profile='invalid_profile',
                strategy='balanced'
            )
        
        # Invalid strategy
        with pytest.raises(ValueError, match="strategy must be one of"):
            MaxCutJobRequest(
                num_nodes=30,
                edge_probability=0.3,
                edge_profile='aerospace',
                strategy='invalid_strategy'
            )
