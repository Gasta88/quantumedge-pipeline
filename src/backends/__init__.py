"""
Pluggable quantum hardware backends.

This package provides a vendor-agnostic interface for quantum hardware,
driven by company profiles. Use create_backend() to instantiate the
correct backend for the active profile.

Usage:
    >>> from src.backends import create_backend, QuantumBackend
    >>> backend = create_backend("photonic_mock")
    >>> job_id = backend.submit_job(circuit, shots=1024)
"""

from src.backends.backend_base import QuantumBackend
from src.backends.photonic_mock import PhotonicMockBackend
from src.backends.photonic_cloud import PhotonicCloudBackend


_BACKEND_REGISTRY = {
    "photonic_mock": PhotonicMockBackend,
    "photonic_cloud": PhotonicCloudBackend,
}


def create_backend(backend_name: str, **kwargs) -> QuantumBackend:
    """
    Factory function to create a quantum backend by name.

    Args:
        backend_name: Identifier matching a profile's hardware_backend field.
                      Supported: 'photonic_mock', 'photonic_cloud'
        **kwargs: Passed to the backend constructor (e.g. api_key for photonic cloud).

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
    "PhotonicMockBackend",
    "PhotonicCloudBackend",
    "create_backend",
]
