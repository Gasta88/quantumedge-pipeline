"""
Pre-configured Demo Scenarios for QuantumEdge Pipeline

This module provides realistic, pre-configured scenarios that demonstrate
the quantum-classical routing capabilities of the QuantumEdge pipeline
in various real-world contexts.

Scenarios:
---------
1. AEROSPACE_ROUTING: UAV path planning under power constraints
2. FINANCIAL_PORTFOLIO: Real-time portfolio rebalancing at the edge
3. ML_GRAPH_PARTITION: Distributed ML training graph partitioning

Each scenario includes:
- Pre-generated problem configuration
- Expected routing decision and reasoning
- Real-world context and application description
- Recommended edge profile and strategy

Usage:
-----
>>> from dashboard.demo_scenarios import DEMO_SCENARIOS, load_demo_scenario
>>> scenario_data = load_demo_scenario('AEROSPACE_ROUTING')
>>> print(scenario_data['description'])
"""

from typing import Dict, Any
from src.problems.maxcut import MaxCutProblem
from src.problems.portfolio import PortfolioOptimizationProblem


# =============================================================================
# Demo Scenario Definitions
# =============================================================================

DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "AEROSPACE_ROUTING": {
        "name": "Aerospace UAV Path Planning",
        "description": "UAV path planning under power constraints for autonomous flight operations",
        "problem_type": "MaxCut",
        "problem_size": 40,
        "problem_params": {
            "edge_probability": 0.25,
            "seed": 42
        },
        "edge_profile": "aerospace",
        "strategy": "energy",
        "comparative_mode": True,
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
                "Graph density allows efficient quantum encoding"
            ]
        }
    },
    
    "FINANCIAL_PORTFOLIO": {
        "name": "Financial Portfolio Optimization",
        "description": "Real-time portfolio rebalancing at the edge for high-frequency trading",
        "problem_type": "Portfolio",
        "problem_size": 50,
        "problem_params": {
            "seed": 123
        },
        "edge_profile": "ground",
        "strategy": "quality",
        "comparative_mode": True,
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
                "Financial models benefit from quantum sampling"
            ]
        }
    },
    
    "ML_GRAPH_PARTITION": {
        "name": "ML Graph Partitioning",
        "description": "Distributed ML training graph partitioning for federated learning",
        "problem_type": "MaxCut",
        "problem_size": 60,
        "problem_params": {
            "edge_probability": 0.4,  # Dense graph
            "seed": 999
        },
        "edge_profile": "mobile",
        "strategy": "balanced",
        "comparative_mode": True,
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
                "Dense graphs may favor classical approximation"
            ]
        }
    }
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
            f"Scenario '{scenario_name}' not found. "
            f"Available: {list(DEMO_SCENARIOS.keys())}"
        )
    
    scenario = DEMO_SCENARIOS[scenario_name].copy()
    
    # Create problem instance based on type
    problem_type = scenario['problem_type']
    problem_size = scenario['problem_size']
    problem_params = scenario['problem_params']
    
    if problem_type == "MaxCut":
        problem = MaxCutProblem(num_nodes=problem_size)
        problem.generate(**problem_params)
    elif problem_type == "Portfolio":
        problem = PortfolioOptimizationProblem(num_assets=problem_size)
        problem.generate(**problem_params)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Add problem instance to scenario
    scenario['problem'] = problem
    
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
**{scenario['name']}**

{scenario['description']}

**Configuration:**
- Problem Type: {scenario['problem_type']}
- Problem Size: {scenario['problem_size']}
- Edge Profile: {scenario['edge_profile']}
- Strategy: {scenario['strategy']}
- Comparative Mode: {'Yes' if scenario['comparative_mode'] else 'No'}

**Expected Routing:**
- Preferred Solver: {scenario['expected_routing']['preferred_solver']}
- Reasoning: {scenario['expected_routing']['reasoning']}

**Context:**
{scenario['context']}
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
    'ground'
    """
    if scenario_name not in DEMO_SCENARIOS:
        raise KeyError(f"Scenario '{scenario_name}' not found.")
    
    scenario = DEMO_SCENARIOS[scenario_name].copy()
    # Remove problem-specific fields that require instantiation
    scenario.pop('problem_params', None)
    
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
        print(f"   Type: {scenario['problem_type']} | "
              f"Size: {scenario['problem_size']} | "
              f"Profile: {scenario['edge_profile']} | "
              f"Strategy: {scenario['strategy']}")
        print()


# =============================================================================
# Module-level Exports
# =============================================================================

__all__ = [
    'DEMO_SCENARIOS',
    'load_demo_scenario',
    'get_scenario_summary',
    'list_available_scenarios',
    'get_scenario_metadata',
    'print_scenario_quick_reference'
]


# =============================================================================
# Command-line Interface
# =============================================================================

if __name__ == "__main__":
    print_scenario_quick_reference()
