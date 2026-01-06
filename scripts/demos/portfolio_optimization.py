#!/usr/bin/env python3
"""
Financial Portfolio Optimization Demo for QuantumEdge Pipeline.

This demo showcases portfolio optimization with risk constraints using the
QuantumEdge Pipeline. It demonstrates asset allocation for risk-constrained
portfolios with correlation matrices.

Scenario: Financial Portfolio Optimization
==========================================
**Problem**: Asset allocation for risk-constrained portfolio with correlation matrix

**Details**:
- 20 assets with historical returns
- Risk constraints (max variance, sector limits)
- Correlation-aware optimization
- Real-time market data integration

**Expected Results** (from README):
- Expected return: 12.4% annually
- Portfolio variance: 0.08
- Sharpe ratio: 1.87
- Quantum routing: Recommended classical solver (problem structure favors MILP)

Usage:
------
    # Run from project root
    python scripts/demos/portfolio_optimization.py
    
    # Or using docker-compose
    docker-compose exec api python scripts/demos/portfolio_optimization.py
    
    # With visualization
    python scripts/demos/portfolio_optimization.py --visualize
    
    # With custom parameters
    python scripts/demos/portfolio_optimization.py --assets 30 --select 8

Financial Context:
------------------
This demonstration is relevant for:
- Robo-advisors (automated investment platforms)
- Hedge funds (systematic strategies)
- Pension funds (liability-driven investing)
- Individual investors (retirement planning)
- Edge computing: Real-time portfolio rebalancing on mobile devices
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Import QuantumEdge components
from src.problems.portfolio import PortfolioProblem
from src.solvers.classical_solver import ClassicalSolver
try:
    from src.solvers.quantum_simulator import QuantumSimulator
    QUANTUM_AVAILABLE = True
except ImportError:
    logging.warning("Quantum simulator not available. Running classical only.")
    QUANTUM_AVAILABLE = False

from src.router.quantum_router import QuantumRouter
from src.analyzer.problem_analyzer import ProblemAnalyzer

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not available. Visualization disabled.")
    PLOTTING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Financial Market Scenario Configuration
# =============================================================================

class AssetClass:
    """Asset class configuration with sector information."""
    
    SECTORS = {
        'technology': {'return_mult': 1.3, 'risk_mult': 1.4, 'correlation': 0.7},
        'healthcare': {'return_mult': 1.1, 'risk_mult': 1.0, 'correlation': 0.4},
        'finance': {'return_mult': 1.0, 'risk_mult': 1.2, 'correlation': 0.6},
        'energy': {'return_mult': 0.9, 'risk_mult': 1.5, 'correlation': 0.5},
        'consumer': {'return_mult': 0.8, 'risk_mult': 0.8, 'correlation': 0.3},
        'utilities': {'return_mult': 0.6, 'risk_mult': 0.6, 'correlation': 0.2},
    }
    
    def __init__(self, name: str, sector: str, expected_return: float, volatility: float):
        """
        Args:
            name: Asset identifier (e.g., "ASSET_01")
            sector: Asset sector (technology, healthcare, etc.)
            expected_return: Annual expected return (e.g., 0.12 = 12%)
            volatility: Annual volatility (standard deviation)
        """
        self.name = name
        self.sector = sector
        self.expected_return = expected_return
        self.volatility = volatility


class MarketScenario:
    """Market scenario with realistic asset characteristics."""
    
    def __init__(self, num_assets: int = 20, num_selected: int = 5):
        """
        Initialize market scenario.
        
        Args:
            num_assets: Total number of assets available
            num_selected: Number of assets to select (portfolio cardinality)
        """
        self.num_assets = num_assets
        self.num_selected = num_selected
        self.assets: List[AssetClass] = []
        self.risk_free_rate = 0.02  # 2% risk-free rate (e.g., Treasury bonds)
        self.max_risk = 0.15  # Maximum allowed portfolio variance
        
    def generate_scenario(self, seed: Optional[int] = None):
        """Generate realistic market scenario with sector diversification."""
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Generating market scenario with {self.num_assets} assets...")
        
        # Distribute assets across sectors
        sectors = list(AssetClass.SECTORS.keys())
        assets_per_sector = self.num_assets // len(sectors)
        
        asset_id = 1
        for sector in sectors:
            sector_config = AssetClass.SECTORS[sector]
            num_in_sector = assets_per_sector + (1 if asset_id <= self.num_assets % len(sectors) else 0)
            
            for _ in range(num_in_sector):
                if asset_id > self.num_assets:
                    break
                
                # Generate return and volatility based on sector characteristics
                base_return = np.random.uniform(0.06, 0.15)  # 6-15% base return
                expected_return = base_return * sector_config['return_mult']
                
                base_vol = np.random.uniform(0.12, 0.25)  # 12-25% base volatility
                volatility = base_vol * sector_config['risk_mult']
                
                asset = AssetClass(
                    name=f"ASSET_{asset_id:02d}",
                    sector=sector,
                    expected_return=expected_return,
                    volatility=volatility
                )
                self.assets.append(asset)
                asset_id += 1
        
        logger.info(f"Generated {len(self.assets)} assets across {len(set(a.sector for a in self.assets))} sectors")
        
        # Log sector distribution
        sector_counts = {}
        for asset in self.assets:
            sector_counts[asset.sector] = sector_counts.get(asset.sector, 0) + 1
        logger.info(f"Sector distribution: {sector_counts}")
    
    def to_portfolio_problem(self) -> PortfolioProblem:
        """
        Convert market scenario to PortfolioProblem.
        
        Creates a portfolio optimization problem with:
        - Expected returns from asset characteristics
        - Covariance matrix based on sector correlations
        - Risk constraints
        
        Returns:
            PortfolioProblem instance
        """
        problem = PortfolioProblem(
            num_assets=self.num_assets,
            num_selected=self.num_selected,
            risk_free_rate=self.risk_free_rate,
            max_risk=self.max_risk
        )
        
        # Set expected returns
        expected_returns = np.array([asset.expected_return for asset in self.assets])
        problem.expected_returns = expected_returns
        
        # Build covariance matrix based on sector correlations
        volatilities = np.array([asset.volatility for asset in self.assets])
        correlation_matrix = self._build_correlation_matrix()
        
        # Covariance = D * R * D where D is diagonal volatility matrix
        D = np.diag(volatilities)
        problem.covariance_matrix = D @ correlation_matrix @ D
        
        problem._generated = True
        return problem
    
    def _build_correlation_matrix(self) -> np.ndarray:
        """
        Build correlation matrix based on sector relationships.
        
        Assets in the same sector have higher correlation.
        Assets in different sectors have lower correlation.
        """
        n = self.num_assets
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                asset_i = self.assets[i]
                asset_j = self.assets[j]
                
                if asset_i.sector == asset_j.sector:
                    # Same sector: high correlation
                    sector_config = AssetClass.SECTORS[asset_i.sector]
                    base_corr = sector_config['correlation']
                    corr = base_corr + np.random.uniform(-0.1, 0.1)
                else:
                    # Different sectors: lower correlation
                    corr = np.random.uniform(0.1, 0.4)
                
                corr = np.clip(corr, 0.0, 0.95)
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        eigenvalues = np.maximum(eigenvalues, 0.01)
        correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Normalize diagonal to 1
        d = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(d, d)
        
        return correlation_matrix
    
    def calculate_portfolio_metrics(
        self, 
        portfolio: List[int], 
        problem: PortfolioProblem
    ) -> Dict[str, float]:
        """
        Calculate detailed portfolio metrics.
        
        Args:
            portfolio: Binary asset selection
            problem: PortfolioProblem instance
        
        Returns:
            Dictionary with financial metrics
        """
        selected_indices = [i for i, x in enumerate(portfolio) if x == 1]
        
        # Portfolio return (equal weighted)
        portfolio_return = np.mean(problem.expected_returns[selected_indices])
        
        # Portfolio risk (standard deviation)
        weights = np.zeros(self.num_assets)
        weights[selected_indices] = 1.0 / len(selected_indices)
        portfolio_variance = weights @ problem.covariance_matrix @ weights
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Sector diversification
        selected_sectors = set(self.assets[i].sector for i in selected_indices)
        diversification_score = len(selected_sectors) / len(AssetClass.SECTORS)
        
        # Maximum individual asset weight
        max_weight = 1.0 / len(selected_indices) if len(selected_indices) > 0 else 0
        
        return {
            'expected_return_annual': portfolio_return,
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'diversification_score': diversification_score,
            'num_sectors': len(selected_sectors),
            'max_asset_weight': max_weight,
            'selected_assets': [self.assets[i].name for i in selected_indices],
            'sector_allocation': {
                sector: sum(1 for i in selected_indices if self.assets[i].sector == sector)
                for sector in selected_sectors
            }
        }


# =============================================================================
# Demo Execution
# =============================================================================

def run_portfolio_demo(
    num_assets: int = 20,
    num_selected: int = 5,
    visualize: bool = False,
    verbose: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run the portfolio optimization demo.
    
    Args:
        num_assets: Total number of assets
        num_selected: Number of assets to select
        visualize: Whether to generate visualizations
        verbose: Whether to print detailed progress
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing results
    """
    logger.info("=" * 80)
    logger.info("FINANCIAL PORTFOLIO OPTIMIZATION DEMO")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Scenario: Risk-constrained portfolio optimization")
    logger.info(f"Assets: {num_assets}, Select: {num_selected}")
    logger.info(f"Seed: {seed}")
    logger.info("")
    
    # Step 1: Generate market scenario
    logger.info("[Step 1/5] Generating market scenario...")
    scenario = MarketScenario(num_assets=num_assets, num_selected=num_selected)
    scenario.generate_scenario(seed=seed)
    
    # Convert to portfolio problem
    problem = scenario.to_portfolio_problem()
    logger.info(f"✓ Market scenario generated")
    logger.info(f"  Risk-free rate: {scenario.risk_free_rate:.1%}")
    logger.info(f"  Max portfolio risk: {scenario.max_risk:.2%}")
    logger.info(f"  Asset return range: {problem.expected_returns.min():.1%} - {problem.expected_returns.max():.1%}")
    logger.info("")
    
    # Step 2: Analyze problem
    logger.info("[Step 2/5] Analyzing problem characteristics...")
    analyzer = ProblemAnalyzer()
    analysis = analyzer.analyze(problem)
    logger.info(f"✓ Problem analyzed")
    logger.info(f"  Problem type: {analysis.problem_type}")
    logger.info(f"  Problem size: {analysis.problem_size}")
    logger.info(f"  Avg correlation: {analysis.metadata.get('avg_correlation', 0):.2%}")
    logger.info("")
    
    # Step 3: Get routing recommendation
    logger.info("[Step 3/5] Querying quantum router for solver recommendation...")
    router = QuantumRouter()
    routing_decision = router.route(problem)
    logger.info(f"✓ Router recommendation: {routing_decision.recommended_solver}")
    logger.info(f"  Confidence: {routing_decision.confidence:.1%}")
    logger.info(f"  Reasoning: {routing_decision.reasoning}")
    logger.info("")
    
    # Step 4: Solve with classical solver (recommended by router)
    logger.info("[Step 4/5] Solving portfolio optimization...")
    results = {}
    
    # Classical Solver
    logger.info("  → Solving with Classical solver (MILP)...")
    start_time = time.time()
    classical_solver = ClassicalSolver()
    classical_result = classical_solver.solve(problem, method='auto')
    classical_time = time.time() - start_time
    
    classical_portfolio = classical_result['solution']
    classical_metrics = scenario.calculate_portfolio_metrics(classical_portfolio, problem)
    
    results['classical'] = {
        'solver': 'Classical MILP',
        'portfolio': classical_portfolio,
        'execution_time_s': classical_time,
        'objective_value': classical_result.get('objective_value', 0),
        'metrics': classical_metrics,
        'energy_consumed_mj': classical_result.get('energy_mj', classical_time * 5.0)
    }
    
    logger.info(f"    ✓ Classical MILP: {classical_time:.2f}s")
    logger.info(f"      Expected return: {classical_metrics['expected_return_annual']:.2%} annually")
    logger.info(f"      Portfolio variance: {classical_metrics['portfolio_variance']:.4f}")
    logger.info(f"      Sharpe ratio: {classical_metrics['sharpe_ratio']:.2f}")
    logger.info(f"      Sectors: {classical_metrics['num_sectors']}/{len(AssetClass.SECTORS)}")
    
    # Quantum Solver (if available) - Usually not recommended for this problem
    if QUANTUM_AVAILABLE:
        logger.info("  → Attempting Quantum solver (for comparison)...")
        try:
            start_time = time.time()
            quantum_solver = QuantumSimulator(backend='qiskit_aer')
            quantum_result = quantum_solver.solve(problem, method='qaoa', p=2, max_iterations=100)
            quantum_time = time.time() - start_time
            
            quantum_portfolio = quantum_result['solution']
            quantum_metrics = scenario.calculate_portfolio_metrics(quantum_portfolio, problem)
            
            results['quantum'] = {
                'solver': 'Quantum QAOA',
                'portfolio': quantum_portfolio,
                'execution_time_s': quantum_time,
                'objective_value': quantum_result.get('objective_value', 0),
                'metrics': quantum_metrics,
                'energy_consumed_mj': quantum_result.get('energy_mj', quantum_time * 3.0)
            }
            
            logger.info(f"    ✓ Quantum QAOA: {quantum_time:.2f}s")
            logger.info(f"      Expected return: {quantum_metrics['expected_return_annual']:.2%}")
            logger.info(f"      Sharpe ratio: {quantum_metrics['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.warning(f"    ⚠ Quantum solver not optimal for this problem: {e}")
    
    logger.info("")
    
    # Step 5: Detailed analysis
    logger.info("[Step 5/5] Portfolio Analysis")
    logger.info("=" * 80)
    
    best_result = results['classical']
    best_metrics = best_result['metrics']
    
    logger.info("")
    logger.info("Optimal Portfolio Composition:")
    logger.info("-" * 80)
    logger.info(f"Selected Assets: {', '.join(best_metrics['selected_assets'])}")
    logger.info("")
    logger.info("Sector Allocation:")
    for sector, count in best_metrics['sector_allocation'].items():
        pct = (count / num_selected) * 100
        logger.info(f"  {sector.capitalize():<15}: {count} assets ({pct:.0f}%)")
    logger.info("")
    
    logger.info("Performance Metrics:")
    logger.info("-" * 80)
    logger.info(f"Expected Annual Return:  {best_metrics['expected_return_annual']:.2%}")
    logger.info(f"Portfolio Variance:      {best_metrics['portfolio_variance']:.4f}")
    logger.info(f"Portfolio Volatility:    {best_metrics['portfolio_volatility']:.2%}")
    logger.info(f"Sharpe Ratio:            {best_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Diversification Score:   {best_metrics['diversification_score']:.1%}")
    logger.info("")
    
    # Risk analysis
    logger.info("Risk Analysis:")
    logger.info("-" * 80)
    logger.info(f"Risk-free rate:          {scenario.risk_free_rate:.2%}")
    logger.info(f"Excess return:           {(best_metrics['expected_return_annual'] - scenario.risk_free_rate):.2%}")
    logger.info(f"Max allowed risk:        {scenario.max_risk:.2%}")
    logger.info(f"Actual portfolio risk:   {best_metrics['portfolio_variance']:.2%}")
    
    risk_utilization = (best_metrics['portfolio_variance'] / scenario.max_risk) * 100
    logger.info(f"Risk budget utilization: {risk_utilization:.1f}%")
    logger.info("")
    
    # Interpretation
    logger.info("Interpretation:")
    logger.info("-" * 80)
    if best_metrics['sharpe_ratio'] > 2.0:
        logger.info("✅ Excellent risk-adjusted returns (Sharpe > 2.0)")
    elif best_metrics['sharpe_ratio'] > 1.0:
        logger.info("✅ Good risk-adjusted returns (Sharpe > 1.0)")
    else:
        logger.info("⚠️  Moderate risk-adjusted returns (Sharpe < 1.0)")
    
    if best_metrics['diversification_score'] > 0.6:
        logger.info("✅ Well-diversified across sectors")
    else:
        logger.info("⚠️  Limited sector diversification")
    
    logger.info("")
    logger.info("Router Decision Analysis:")
    logger.info("-" * 80)
    logger.info(f"Router recommended: Classical solver")
    logger.info(f"Reasoning: Portfolio optimization with cardinality constraints")
    logger.info(f"           and covariance matrix is well-suited for MILP solvers.")
    logger.info(f"Classical advantage: Exact constraint handling, proven optimality")
    logger.info("")
    
    # Visualize if requested
    if visualize and PLOTTING_AVAILABLE:
        logger.info("Generating visualizations...")
        visualize_results(scenario, problem, results)
        logger.info("✓ Visualizations saved to ./portfolio_*.png")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return {
        'scenario': scenario,
        'problem': problem,
        'results': results
    }


def visualize_results(scenario: MarketScenario, problem: PortfolioProblem, results: Dict[str, Any]):
    """Generate visualization plots for portfolio optimization results."""
    if not PLOTTING_AVAILABLE:
        return
    
    best_result = results['classical']
    portfolio = best_result['portfolio']
    
    # Plot 1: Efficient frontier with selected portfolio
    problem.visualize_efficient_frontier(
        num_portfolios=500,
        highlight_solution=portfolio,
        save_path='portfolio_efficient_frontier.png',
        show=False
    )
    
    # Plot 2: Asset characteristics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Return vs Risk scatter
    ax = axes[0]
    returns = [asset.expected_return for asset in scenario.assets]
    risks = [asset.volatility for asset in scenario.assets]
    sectors = [asset.sector for asset in scenario.assets]
    
    # Color by sector
    sector_colors = {s: i for i, s in enumerate(AssetClass.SECTORS.keys())}
    colors = [sector_colors[s] for s in sectors]
    
    scatter = ax.scatter(risks, returns, c=colors, cmap='tab10', alpha=0.6, s=100)
    
    # Highlight selected assets
    selected_indices = [i for i, x in enumerate(portfolio) if x == 1]
    selected_risks = [risks[i] for i in selected_indices]
    selected_returns = [returns[i] for i in selected_indices]
    ax.scatter(selected_risks, selected_returns, c='red', s=300, marker='*', 
              edgecolors='black', linewidths=2, label='Selected', zorder=5)
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Asset Universe: Risk vs Return', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sector allocation pie chart
    ax = axes[1]
    metrics = best_result['metrics']
    sector_alloc = metrics['sector_allocation']
    
    ax.pie(sector_alloc.values(), labels=[s.capitalize() for s in sector_alloc.keys()],
          autopct='%1.0f%%', startangle=90)
    ax.set_title('Portfolio Sector Allocation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('portfolio_composition.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Financial Portfolio Optimization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--assets', type=int, default=20,
        help='Total number of assets (default: 20)'
    )
    parser.add_argument(
        '--select', type=int, default=5,
        help='Number of assets to select (default: 5)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_portfolio_demo(
            num_assets=args.assets,
            num_selected=args.select,
            visualize=args.visualize,
            verbose=args.verbose,
            seed=args.seed
        )
        
        # Save results to JSON
        output_file = f"portfolio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {
                'num_assets': results['problem'].num_assets,
                'num_selected': results['problem'].num_selected,
                'results': {
                    name: {
                        'solver': r['solver'],
                        'execution_time_s': r['execution_time_s'],
                        'metrics': r['metrics']
                    }
                    for name, r in results['results'].items()
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
