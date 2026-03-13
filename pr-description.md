## What Changed
- introduce Target Profiles architecture with profile loader, YAML configs (`profiles/`), and dual Rotonium/QuiX scenarios plus docs
- add pluggable `QuantumBackend` interface with Rotonium mock + QuiX cloud backends, and new datacenter simulator/JSON scenarios wired into dashboard & API
- update infra/docs/deps (docker-compose, pyproject, README, quix-integration doc) and add dedicated tests for profiles, backends, and datacenter routing

## Why This Change
- enables the same codebase to pitch both Rotonium (edge) and QuiX (data center) hardware by swapping declarative profiles instead of forking features

## Testing Done
- `poetry run pytest tests/ -v --tb=short --no-header` *(fails: known pre-existing failure in tests/test_solvers.py::TestPortfolioSolver::test_portfolio_scipy_sharpe_success – unrelated to this work)*
