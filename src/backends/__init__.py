"""
Pluggable quantum hardware backends.

This package provides a vendor-agnostic interface for quantum hardware,
driven by company profiles. Use create_backend() to instantiate the
correct backend for the active profile.

Usage:
    >>> from src.backends import create_backend, QuantumBackend
    >>> backend = create_backend("rotonium_mock")
    >>> job_id = backend.submit_job(circuit, shots=1024)
"""

from src.backends.backend_base import QuantumBackend
from src.backends.rotonium_mock import RotoniumMockBackend
from src.backends.quix_cloud import QuiXCloudBackend


_BACKEND_REGISTRY = {
    "rotonium_mock": RotoniumMockBackend,
    "quix_cloud": QuiXCloudBackend,
}


def create_backend(backend_name: str, **kwargs) -> QuantumBackend:
    """
    Factory function to create a quantum backend by name.

    Args:
        backend_name: Identifier matching a profile's hardware_backend field.
                      Supported: 'rotonium_mock', 'quix_cloud'
        **kwargs: Passed to the backend constructor (e.g. api_key for QuiX).

    Returns:
        An instance of the requested QuantumBackend implementation.

    Raises:
        ValueError: If backend_name is not recognised.
    """
    if backend_name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available: {list(_BACKEND_REGISTRY.keys())}"
        )
    return _BACKEND_REGISTRY[backend_name](**kwargs)


__all__ = [
    "QuantumBackend",
    "RotoniumMockBackend",
    "QuiXCloudBackend",
    "create_backend",
]
