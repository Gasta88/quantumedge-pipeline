"""
Pre-configured Demo Scenarios for QuantumEdge Pipeline

This module provides realistic, pre-configured scenarios that demonstrate
the quantum-classical routing capabilities of the QuantumEdge pipeline
in various real-world contexts.

Scenarios are split by company profile:

Rotonium (edge):
    1. AEROSPACE_ROUTING: UAV path planning under power constraints
    2. FINANCIAL_PORTFOLIO: Real-time portfolio rebalancing at the edge
    3. ML_GRAPH_PARTITION: Distributed ML training graph partitioning

QuiX Quantum (datacenter):
    1. PHARMA_OPTIMIZATION: Drug discovery molecular sampling
    2. PORTFOLIO_RISK: Financial risk modelling
    3. HYDROLOGY: Water-network flow optimisation

Each scenario includes:
- Pre-generated problem configuration
- Expected routing decision and reasoning
- Real-world context and application description
- Recommended edge/deployment profile and strategy

Usage:
-----
>>> from dashboard.demo_scenarios import get_scenarios_for_profile, load_demo_scenario
>>> scenarios = get_scenarios_for_profile("rotonium")
>>> scenario_data = load_demo_scenario('AEROSPACE_ROUTING')
>>> print(scenario_data['description'])
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.problems.maxcut import MaxCutProblem
from src.problems.tsp import TSPProblem
from src.problems.portfolio import PortfolioProblem

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Demo Scenario Definitions
# =============================================================================

DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "AEROSPACE_ROUTING": {
        "name": "Aerospace UAV Path Planning",
        "description": "UAV path planning under power constraints for autonomous flight operations",
        "problem_type": "MaxCut",
        "problem_size": 40,
        "problem_params": {"edge_probability": 0.25, "seed": 42},
        "edge_profile": "aerospace",
        "strategy": "energy",
        "comparative_mode": True,
        "company_profile": "rotonium",
        "context": """
        **Real-World Application: Autonomous UAV Fleet Coordination**
        
        Unmanned Aerial Vehicles (UAVs) operating in swarms need to efficiently
        partition airspace for patrol missions while minimizing power consumption.
        Each UAV has limited battery capacity (~50W peak power budget).
        
        The MaxCut problem models the optimal separation of UAV clusters to:
        - Minimize communication overhead between groups
        - Maximize coverage efficiency
        - Operate within strict power constraints
        
        **Why This Matters:**
        - UAVs have limited onboard compute (edge devices)
        - Energy-efficient algorithms extend flight time
        - Real-time decision-making is critical for safety
        
        **Expected Outcome:**
        Quantum solver may offer energy advantages for this problem size,
        making it viable for battery-powered edge deployment.
        """,
        "expected_routing": {
            "preferred_solver": "quantum",
            "reasoning": "Energy-optimized strategy with power constraints favors quantum for this size",
            "key_factors": [
                "Problem size (40 nodes) is in quantum advantage zone",
                "Aerospace power budget (50W) is tight",
                "Energy strategy prioritizes lower power consumption",
                "Graph density allows efficient quantum encoding",
            ],
        },
    },
    "FINANCIAL_PORTFOLIO": {
        "name": "Financial Portfolio Optimization",
        "description": "Real-time portfolio rebalancing at the edge for high-frequency trading",
        "problem_type": "Portfolio",
        "problem_size": 50,
        "problem_params": {"seed": 123},
        "edge_profile": "ground_server",
        "strategy": "quality",
        "comparative_mode": True,
        "company_profile": "rotonium",
        "context": """
        **Real-World Application: Edge-Based Trading Systems**
        
        High-frequency trading firms deploy edge servers near exchanges to minimize
        latency. Portfolio optimization must happen in real-time as market conditions
        change rapidly throughout the trading day.
        
        The portfolio optimization problem balances:
        - Expected returns vs risk (Sharpe ratio maximization)
        - Diversification constraints
        - Transaction costs
        - Computational budget (must complete within milliseconds)
        
        **Why This Matters:**
        - Every millisecond of latency costs money in HFT
        - Solution quality directly impacts profitability
        - Edge deployment reduces network round-trip time
        
        **Expected Outcome:**
        For 50 assets, quantum approaches may provide better solution quality
        even if slightly slower, as the quality-optimized strategy prioritizes
        finding superior portfolios.
        """,
        "expected_routing": {
            "preferred_solver": "quantum",
            "reasoning": "Quality-focused strategy with ample power budget favors quantum exploration",
            "key_factors": [
                "Ground server profile has high power budget (200W)",
                "Quality strategy prioritizes solution superiority",
                "50 assets is moderate size for quantum",
                "Financial models benefit from quantum sampling",
            ],
        },
    },
    "ML_GRAPH_PARTITION": {
        "name": "ML Graph Partitioning",
        "description": "Distributed ML training graph partitioning for federated learning",
        "problem_type": "MaxCut",
        "problem_size": 60,
        "problem_params": {
            "edge_probability": 0.4,  # Dense graph
            "seed": 999,
        },
        "edge_profile": "mobile",
        "strategy": "balanced",
        "comparative_mode": True,
        "company_profile": "rotonium",
        "context": """
        **Real-World Application: Federated ML Training**
        
        Distributed machine learning systems partition computational graphs across
        mobile edge devices for privacy-preserving federated learning. The goal is
        to minimize communication between partitions while balancing compute load.
        
        The MaxCut formulation optimizes:
        - Communication minimization (cut edges = data transfer)
        - Load balancing across mobile devices
        - Partition size constraints
        - Energy efficiency on battery-powered devices
        
        **Why This Matters:**
        - Mobile devices have limited power (5-15W budget)
        - Communication is expensive in terms of energy and latency
        - Privacy regulations require edge processing
        - Graph density reflects interconnected neural network layers
        
        **Expected Outcome:**
        Balanced strategy on mobile profile will make nuanced trade-off between
        quantum quality benefits and classical speed advantages.
        """,
        "expected_routing": {
            "preferred_solver": "mixed",
            "reasoning": "Balanced strategy weighs multiple factors; decision depends on runtime analysis",
            "key_factors": [
                "60 nodes with 40% density is challenging for both solvers",
                "Mobile power budget (5-15W) is restrictive",
                "Balanced strategy considers time, energy, and quality equally",
                "Dense graphs may favor classical approximation",
            ],
        },
    },
    # =========================================================================
    # QuiX Quantum Scenarios (Data Center)
    # =========================================================================
    "PHARMA_OPTIMIZATION": {
        "name": "Drug Discovery Sampling",
        "description": "Molecular conformation sampling for pharmaceutical lead optimisation",
        "problem_type": "MaxCut",
        "problem_size": 50,
        "problem_params": {"edge_probability": 0.3, "seed": 200},
        "edge_profile": "ground_server",
        "strategy": "quality",
        "comparative_mode": True,
        "company_profile": "quix",
        "context": """
        **Real-World Application: Drug Discovery with QuiX Quantum**

        Pharmaceutical companies need to explore molecular energy landscapes
        efficiently to identify promising drug candidates. The MaxCut formulation
        maps molecular interaction graphs to find optimal conformations.

        QuiX Quantum's silicon-nitride photonic processor provides:
        - High-fidelity quantum sampling (>99% circuit fidelity)
        - Data-center integration via standard rack mounting
        - No cryogenic overhead — lower total cost of ownership

        **Expected Outcome:**
        Quality-optimised strategy on HPC cluster leverages quantum sampling
        to explore solution space more thoroughly than classical heuristics.
        """,
        "expected_routing": {
            "preferred_solver": "quantum",
            "reasoning": "Quality strategy with ample HPC resources favours quantum exploration",
            "key_factors": [
                "HPC cluster has high power budget (5000W)",
                "Quality strategy prioritises solution superiority",
                "50 nodes is suitable for quantum advantage zone",
                "Pharma applications benefit from quantum sampling diversity",
            ],
        },
    },
    "PORTFOLIO_RISK": {
        "name": "Financial Risk Modeling",
        "description": "Portfolio risk assessment and VaR estimation using quantum sampling",
        "problem_type": "Portfolio",
        "problem_size": 40,
        "problem_params": {"seed": 300},
        "edge_profile": "ground_server",
        "strategy": "balanced",
        "comparative_mode": True,
        "company_profile": "quix",
        "context": """
        **Real-World Application: Financial Risk with QuiX Quantum**

        Financial institutions model portfolio risk by sampling correlated asset
        returns. Quantum sampling on QuiX's photonic processor provides potential
        quadratic speed-up for Monte Carlo methods.

        Running from a standard data-center rack, the QuiX quantum blade
        integrates alongside existing risk infrastructure with minimal overhead.

        **Expected Outcome:**
        Balanced strategy evaluates quantum sampling vs classical Monte Carlo,
        choosing based on problem size and available datacenter resources.
        """,
        "expected_routing": {
            "preferred_solver": "quantum",
            "reasoning": "Portfolio sampling benefits from quantum advantage in data-center setting",
            "key_factors": [
                "Ground server / datacenter rack has ample resources",
                "40 assets is moderate size for quantum portfolio optimisation",
                "Financial models benefit from diverse quantum sampling",
                "PUE-adjusted energy model tracks data-center efficiency",
            ],
        },
    },
    "HYDROLOGY": {
        "name": "Hydrology Simulation",
        "description": "Water-network flow optimisation for flood risk management",
        "problem_type": "MaxCut",
        "problem_size": 45,
        "problem_params": {"edge_probability": 0.2, "seed": 400},
        "edge_profile": "ground_server",
        "strategy": "balanced",
        "comparative_mode": True,
        "company_profile": "quix",
        "context": """
        **Real-World Application: Hydrology with QuiX Quantum**

        Deltares and QuiX Quantum collaborate on quantum modelling solutions
        for water management and infrastructure. The MaxCut formulation
        partitions drainage networks to optimise pump-station activation.

        Cloud-hosted QuiX quantum instances enable on-demand access to
        photonic quantum hardware for research simulations.

        **Expected Outcome:**
        Balanced strategy on cloud node evaluates cost/quality trade-off,
        leveraging QuiX cloud API for real quantum hardware results.
        """,
        "expected_routing": {
            "preferred_solver": "mixed",
            "reasoning": "Balanced strategy weighs quantum quality against cloud execution cost",
            "key_factors": [
                "Cloud node has moderate resource constraints",
                "45 nodes with sparse graph favours quantum approaches",
                "Real-world collaboration between Deltares and QuiX",
                "PUE model accounts for cloud infrastructure overhead",
            ],
        },
    },
}


# =============================================================================
# Scenario Loading Functions
# =============================================================================


def load_demo_scenario(scenario_name: str) -> Dict[str, Any]:
    """
    Load a pre-configured demo scenario.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario to load (from DEMO_SCENARIOS keys)

    Returns
    -------
    Dict[str, Any]
        Scenario configuration with problem, settings, and context

    Raises
    ------
    KeyError
        If scenario_name is not found in DEMO_SCENARIOS

    Example
    -------
    >>> scenario = load_demo_scenario('AEROSPACE_ROUTING')
    >>> print(scenario['description'])
    >>> problem = scenario['problem']
    >>> edge_profile = scenario['edge_profile']
    """
    if scenario_name not in DEMO_SCENARIOS:
        raise KeyError(
            f"Scenario '{scenario_name}' not found. Available: {list(DEMO_SCENARIOS.keys())}"
        )

    scenario = DEMO_SCENARIOS[scenario_name].copy()

    # Create problem instance based on type
    problem_type = scenario["problem_type"]
    problem_size = scenario["problem_size"]
    problem_params = scenario["problem_params"]

    if problem_type == "MaxCut":
        problem = MaxCutProblem(num_nodes=problem_size)
        problem.generate(**problem_params)
    elif problem_type == "TSP":
        problem = TSPProblem(num_cities=problem_size)
        problem.generate(**problem_params)
    elif problem_type == "Portfolio":
        problem = PortfolioProblem(num_assets=problem_size, num_selected=problem_size // 2)
        problem.generate(**{k: v for k, v in problem_params.items() if k != "euclidean"})
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Add problem instance to scenario
    scenario["problem"] = problem

    return scenario


def get_scenario_summary(scenario_name: str) -> str:
    """
    Get a human-readable summary of a scenario.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario

    Returns
    -------
    str
        Formatted summary text

    Example
    -------
    >>> summary = get_scenario_summary('AEROSPACE_ROUTING')
    >>> print(summary)
    """
    if scenario_name not in DEMO_SCENARIOS:
        return f"Scenario '{scenario_name}' not found."

    scenario = DEMO_SCENARIOS[scenario_name]

    summary = f"""
**{scenario["name"]}**

{scenario["description"]}

**Configuration:**
- Problem Type: {scenario["problem_type"]}
- Problem Size: {scenario["problem_size"]}
- Edge Profile: {scenario["edge_profile"]}
- Strategy: {scenario["strategy"]}
- Comparative Mode: {"Yes" if scenario["comparative_mode"] else "No"}

**Expected Routing:**
- Preferred Solver: {scenario["expected_routing"]["preferred_solver"]}
- Reasoning: {scenario["expected_routing"]["reasoning"]}

**Context:**
{scenario["context"]}
    """

    return summary.strip()


def list_available_scenarios() -> list:
    """
    Get list of all available scenario names.

    Returns
    -------
    list
        List of scenario names

    Example
    -------
    >>> scenarios = list_available_scenarios()
    >>> for name in scenarios:
    ...     print(f"- {name}")
    """
    return list(DEMO_SCENARIOS.keys())


def get_scenario_metadata(scenario_name: str) -> Dict[str, Any]:
    """
    Get metadata for a scenario without creating the problem instance.

    Useful for displaying scenario information without the computational
    cost of generating the problem.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario

    Returns
    -------
    Dict[str, Any]
        Scenario metadata (excluding problem instance)

    Example
    -------
    >>> metadata = get_scenario_metadata('FINANCIAL_PORTFOLIO')
    >>> print(metadata['edge_profile'])
    'ground_server'
    """
    if scenario_name not in DEMO_SCENARIOS:
        raise KeyError(f"Scenario '{scenario_name}' not found.")

    scenario = DEMO_SCENARIOS[scenario_name].copy()
    # Remove problem-specific fields that require instantiation
    scenario.pop("problem_params", None)

    return scenario


# =============================================================================
# Quick Reference Display
# =============================================================================


def print_scenario_quick_reference():
    """
    Print a quick reference guide for all scenarios.

    Useful for command-line exploration or documentation.

    Example
    -------
    >>> print_scenario_quick_reference()
    """
    print("=" * 80)
    print("QuantumEdge Pipeline - Demo Scenarios Quick Reference")
    print("=" * 80)
    print()

    for name, scenario in DEMO_SCENARIOS.items():
        print(f"📌 {name}")
        print(f"   {scenario['description']}")
        print(
            f"   Type: {scenario['problem_type']} | "
            f"Size: {scenario['problem_size']} | "
            f"Profile: {scenario['edge_profile']} | "
            f"Strategy: {scenario['strategy']}"
        )
        print()


# =============================================================================
# Profile-Aware Scenario Functions
# =============================================================================


def get_scenarios_for_profile(profile_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Return only the demo scenarios that match a given company profile.

    Args:
        profile_name: 'rotonium', 'quix', or any custom profile name.
                      Case-insensitive partial match is used (e.g. 'quix'
                      matches company_profile='quix').

    Returns:
        Filtered dict of scenario_name -> scenario_config.
    """
    profile_lower = profile_name.lower()
    return {
        name: scenario
        for name, scenario in DEMO_SCENARIOS.items()
        if scenario.get("company_profile", "rotonium").lower() == profile_lower
    }


def load_scenario_from_json(json_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a scenario from a JSON file referenced in a profile YAML.

    Args:
        json_path: Relative path from project root, e.g.
                   'examples/scenarios/quix/pharma_optimization.json'

    Returns:
        Parsed scenario dict, or None if file not found.
    """
    full_path = PROJECT_ROOT / json_path
    if not full_path.exists():
        logger.warning(f"Scenario file not found: {full_path}")
        return None
    with open(full_path, "r") as f:
        return json.load(f)


# =============================================================================
# Module-level Exports
# =============================================================================

__all__ = [
    "DEMO_SCENARIOS",
    "load_demo_scenario",
    "get_scenario_summary",
    "list_available_scenarios",
    "get_scenario_metadata",
    "get_scenarios_for_profile",
    "load_scenario_from_json",
    "print_scenario_quick_reference",
]


# =============================================================================
# Command-line Interface
# =============================================================================

if __name__ == "__main__":
    print_scenario_quick_reference()
