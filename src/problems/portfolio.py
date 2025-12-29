"""
Portfolio Optimization Problem Implementation.

Portfolio optimization is a fundamental problem in quantitative finance,
seeking to balance expected returns against risk (variance) when selecting
assets for investment.

Problem Definition:
    Given n assets with expected returns μᵢ and covariance matrix Σ,
    select a subset of k assets to maximize the risk-adjusted return.
    
    Mean-Variance Framework (Markowitz, 1952):
    -------------------------------------------
    Portfolio return:  R = Σᵢ wᵢ μᵢ
    Portfolio risk:    σ² = w^T Σ w
    
    where:
    - wᵢ ∈ [0, 1]: weight of asset i in portfolio
    - Σᵢ wᵢ = 1: weights sum to 100%
    - μᵢ: expected return of asset i
    - Σ: covariance matrix (captures correlations)
    - σ²: portfolio variance (risk measure)

Simplified Binary Formulation:
    In this implementation, we use binary asset selection (wᵢ ∈ {0, 1}):
    - wᵢ = 1: asset i is included
    - wᵢ = 0: asset i is excluded
    - Equal weighting: each selected asset gets weight 1/k
    
    Objective: Maximize Sharpe ratio = (Return - Risk_free_rate) / Risk
    
    Constraints:
    - Cardinality: Select exactly k assets (budget/diversification limit)
    - Risk bound: Portfolio variance ≤ max_risk

Complexity:
    - NP-hard when cardinality constraints are included
    - Combinatorial: C(n, k) = n!/(k!(n-k)!) possible portfolios
    - Example: Selecting 10 assets from 50 gives ~10 billion combinations
    
    Without cardinality constraints:
    - Continuous mean-variance: Convex optimization (polynomial time)
    - Binary selection with cardinality: NP-hard

Financial Context:
    Portfolio optimization is crucial for:
    - Robo-advisors (automated investment platforms)
    - Hedge funds (systematic strategies)
    - Pension funds (liability-driven investing)
    - Individual investors (retirement planning)
    
    Edge computing relevance:
    - Real-time portfolio rebalancing on mobile devices
    - High-frequency trading with low-latency constraints
    - Decentralized finance (DeFi) applications
    - IoT-enabled financial services

Quantum Advantage Potential:
    Portfolio optimization with cardinality constraints is well-suited
    for quantum optimization:
    - Natural binary encoding (asset selection)
    - Quadratic objective (covariance matrix → QUBO)
    - Moderate problem sizes (50-200 assets typical)
    - Real-world financial impact

Financial Terms Glossary:
    - Expected Return (μ): Mean historical or predicted asset return
    - Variance (σ²): Measure of return volatility (risk)
    - Covariance (Σᵢⱼ): How two assets move together
    - Correlation (ρᵢⱼ): Normalized covariance (-1 to 1)
    - Sharpe Ratio: Risk-adjusted return metric
    - Efficient Frontier: Optimal risk-return tradeoff curve
    - Diversification: Spreading investment across uncorrelated assets
    - Cardinality: Number of assets in portfolio (for simplicity/cost)

Example Usage:
    >>> from src.problems.portfolio import PortfolioProblem
    >>> 
    >>> # Create portfolio with 20 assets, select 5
    >>> problem = PortfolioProblem(num_assets=20, num_selected=5)
    >>> 
    >>> # Generate random returns and covariance
    >>> problem.generate(
    ...     seed=42,
    ...     return_range=(0.05, 0.20),  # 5-20% annual return
    ...     risk_range=(0.10, 0.30)      # 10-30% annual volatility
    ... )
    >>> 
    >>> # Get a random portfolio
    >>> portfolio = problem.get_random_solution(seed=123)
    >>> 
    >>> # Evaluate performance
    >>> cost = problem.calculate_cost(portfolio)
    >>> sharpe = -cost  # Negative because we minimize
    >>> print(f"Sharpe ratio: {sharpe:.3f}")
    >>> 
    >>> # Visualize efficient frontier
    >>> problem.visualize_efficient_frontier(
    ...     num_portfolios=1000,
    ...     highlight_solution=portfolio
    ... )
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize

from src.problems.problem_base import ProblemBase


class PortfolioProblem(ProblemBase):
    """
    Portfolio optimization with cardinality constraints.
    
    Implements binary portfolio selection where assets are either fully
    included (equal weight) or excluded. This is a simplified but practical
    formulation suitable for quantum optimization.
    
    Attributes:
        num_assets (int): Total number of available assets
        num_selected (int): Number of assets to select (cardinality)
        expected_returns (np.ndarray): Expected return for each asset
        covariance_matrix (np.ndarray): Asset covariance matrix
        risk_free_rate (float): Risk-free return rate (for Sharpe ratio)
        max_risk (float): Maximum allowed portfolio variance
    
    Example:
        >>> # Conservative portfolio: 10 from 30, low risk
        >>> problem = PortfolioProblem(num_assets=30, num_selected=10)
        >>> problem.generate(
        ...     seed=42,
        ...     return_range=(0.04, 0.12),  # 4-12% return
        ...     risk_range=(0.08, 0.20),     # 8-20% volatility
        ...     correlation_strength=0.3     # Moderate correlation
        ... )
        >>> 
        >>> # Find best portfolio
        >>> best_portfolio, best_sharpe = problem.get_optimal_solution_brute_force()
        >>> print(f"Optimal Sharpe ratio: {best_sharpe:.3f}")
    """
    
    def __init__(
        self,
        num_assets: int,
        num_selected: int,
        risk_free_rate: float = 0.02,
        max_risk: Optional[float] = None,
    ):
        """
        Initialize portfolio optimization problem.
        
        Args:
            num_assets: Total number of assets to choose from (n)
            num_selected: Number of assets to select (k)
            risk_free_rate: Annual risk-free rate (default: 2% = 0.02)
                           Used in Sharpe ratio calculation
            max_risk: Maximum allowed portfolio variance (optional)
                     If None, no risk constraint
        
        Raises:
            ValueError: If parameters are invalid
        
        Example:
            >>> # Select 5 assets from 15, risk-free rate 3%
            >>> problem = PortfolioProblem(
            ...     num_assets=15,
            ...     num_selected=5,
            ...     risk_free_rate=0.03
            ... )
        """
        super().__init__()
        
        if num_assets < 2:
            raise ValueError("num_assets must be at least 2")
        
        if num_selected < 1 or num_selected > num_assets:
            raise ValueError(f"num_selected must be in [1, {num_assets}]")
        
        if risk_free_rate < 0:
            raise ValueError("risk_free_rate must be non-negative")
        
        self._problem_type = "portfolio"
        self._problem_size = num_assets
        self._complexity_class = "NP-hard"  # With cardinality constraints
        
        self.num_assets = num_assets
        self.num_selected = num_selected
        self.risk_free_rate = risk_free_rate
        self.max_risk = max_risk
        
        self.expected_returns: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
    
    def generate(
        self,
        seed: Optional[int] = None,
        return_range: Tuple[float, float] = (0.05, 0.20),
        risk_range: Tuple[float, float] = (0.10, 0.30),
        correlation_strength: float = 0.5,
        **kwargs
    ) -> None:
        """
        Generate random asset returns and covariance matrix.
        
        Creates synthetic financial data with specified characteristics:
        - Expected returns sampled from uniform distribution
        - Volatilities (standard deviations) sampled uniformly
        - Correlation matrix with controllable average correlation
        - Covariance matrix constructed from volatilities and correlations
        
        Financial Interpretation:
        ------------------------
        - Higher returns typically come with higher risk
        - Correlation strength controls diversification benefit
        - Low correlation: Better diversification (risk reduction)
        - High correlation: Assets move together (less diversification)
        
        Args:
            seed: Random seed for reproducibility
            return_range: (min, max) annual return rates
                         Example: (0.05, 0.20) = 5% to 20% annual return
            risk_range: (min, max) annual volatility (std dev)
                       Example: (0.10, 0.30) = 10% to 30% volatility
            correlation_strength: Average correlation between assets (0 to 1)
                                 0.0: Uncorrelated (best diversification)
                                 0.5: Moderate correlation (typical)
                                 1.0: Perfect correlation (no diversification)
            **kwargs: Additional arguments (ignored)
        
        Raises:
            ValueError: If parameters are invalid
        
        Example:
            >>> problem = PortfolioProblem(num_assets=10, num_selected=3)
            >>> 
            >>> # Conservative assets (lower return, lower risk)
            >>> problem.generate(
            ...     seed=42,
            ...     return_range=(0.03, 0.08),
            ...     risk_range=(0.05, 0.15),
            ...     correlation_strength=0.4
            ... )
            >>> 
            >>> # Aggressive assets (higher return, higher risk)
            >>> problem.generate(
            ...     seed=123,
            ...     return_range=(0.10, 0.30),
            ...     risk_range=(0.20, 0.50),
            ...     correlation_strength=0.6
            ... )
        """
        if return_range[0] >= return_range[1]:
            raise ValueError("return_range must be (min, max) with min < max")
        
        if risk_range[0] <= 0 or risk_range[1] <= risk_range[0]:
            raise ValueError("risk_range must be (min, max) with 0 < min < max")
        
        if not (0.0 <= correlation_strength <= 1.0):
            raise ValueError("correlation_strength must be in [0, 1]")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate expected returns (μ)
        min_return, max_return = return_range
        self.expected_returns = np.random.uniform(
            min_return, max_return, size=self.num_assets
        )
        
        # Generate volatilities (σ - standard deviations)
        min_risk, max_risk = risk_range
        volatilities = np.random.uniform(min_risk, max_risk, size=self.num_assets)
        
        # Generate correlation matrix
        # Start with identity (no correlation)
        correlation_matrix = np.eye(self.num_assets)
        
        # Add random correlations
        for i in range(self.num_assets):
            for j in range(i + 1, self.num_assets):
                # Random correlation around correlation_strength
                corr = np.random.uniform(
                    max(0, correlation_strength - 0.3),
                    min(1, correlation_strength + 0.3)
                )
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr  # Symmetric
        
        # Ensure correlation matrix is positive semi-definite
        # (mathematical requirement for valid covariance matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        eigenvalues = np.maximum(eigenvalues, 0.01)  # Ensure positive
        correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Normalize to ensure diagonal is 1
        d = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(d, d)
        
        # Construct covariance matrix: Σ = D·R·D
        # where D is diagonal matrix of volatilities, R is correlation matrix
        D = np.diag(volatilities)
        self.covariance_matrix = D @ correlation_matrix @ D
        
        self._generated = True
    
    def validate_solution(self, solution: List[int]) -> bool:
        """
        Validate a portfolio selection.
        
        A valid portfolio must:
        1. Have correct length (one decision per asset)
        2. Be binary (0 = not selected, 1 = selected)
        3. Select exactly num_selected assets (cardinality constraint)
        4. Satisfy risk constraint if specified
        
        Args:
            solution: Binary selection [0, 1, 0, 1, ...]
                     solution[i] = 1 means asset i is included
        
        Returns:
            True if portfolio is valid, False otherwise
        
        Example:
            >>> problem = PortfolioProblem(num_assets=10, num_selected=3)
            >>> problem.generate()
            >>> 
            >>> valid = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # 3 assets selected
            >>> assert problem.validate_solution(valid)
            >>> 
            >>> invalid = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 4 assets (wrong count)
            >>> assert not problem.validate_solution(invalid)
        """
        if not self._generated:
            return False
        
        # Check length
        if len(solution) != self.num_assets:
            return False
        
        # Check binary
        if not all(x in [0, 1] for x in solution):
            return False
        
        # Check cardinality (exactly num_selected assets)
        if sum(solution) != self.num_selected:
            return False
        
        # Check risk constraint if specified
        if self.max_risk is not None:
            portfolio_risk = self._calculate_portfolio_risk(solution)
            if portfolio_risk > self.max_risk:
                return False
        
        return True
    
    def calculate_cost(self, solution: List[int]) -> float:
        """
        Calculate negative Sharpe ratio (for minimization).
        
        Sharpe Ratio Definition:
        -----------------------
        Sharpe = (Portfolio Return - Risk-free Rate) / Portfolio Risk
        
        where:
        - Portfolio Return = (1/k) Σᵢ (wᵢ · μᵢ) for selected assets
        - Portfolio Risk = √(w^T Σ w) (standard deviation)
        - k = number of selected assets (equal weighting)
        
        Interpretation:
        - Sharpe > 1: Good risk-adjusted return
        - Sharpe > 2: Very good
        - Sharpe > 3: Excellent
        - Sharpe < 1: Poor risk-adjusted return
        
        Why Sharpe Ratio?
        ----------------
        - Balances return and risk in single metric
        - Widely used in financial industry
        - Allows comparison across different portfolios
        - Penalizes volatility appropriately
        
        Convention: Returns NEGATIVE Sharpe for minimization.
        
        Args:
            solution: Binary portfolio selection
        
        Returns:
            Negative Sharpe ratio (lower is better)
        
        Raises:
            ValueError: If solution is invalid
        
        Example:
            >>> problem = PortfolioProblem(num_assets=10, num_selected=3)
            >>> problem.generate(seed=42)
            >>> portfolio = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            >>> 
            >>> cost = problem.calculate_cost(portfolio)
            >>> sharpe_ratio = -cost
            >>> print(f"Sharpe ratio: {sharpe_ratio:.3f}")
            >>> 
            >>> # Interpret result
            >>> if sharpe_ratio > 1.0:
            ...     print("Good risk-adjusted return")
        """
        if not self.validate_solution(solution):
            raise ValueError("Invalid portfolio solution")
        
        # Calculate portfolio return (equal weighting among selected assets)
        selected_indices = [i for i, x in enumerate(solution) if x == 1]
        portfolio_return = np.mean(self.expected_returns[selected_indices])
        
        # Calculate portfolio risk (standard deviation)
        portfolio_risk = self._calculate_portfolio_risk(solution)
        
        # Sharpe ratio
        excess_return = portfolio_return - self.risk_free_rate
        
        # Avoid division by zero
        if portfolio_risk < 1e-10:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = excess_return / portfolio_risk
        
        # Return negative for minimization
        return -sharpe_ratio
    
    def _calculate_portfolio_risk(self, solution: List[int]) -> float:
        """
        Calculate portfolio variance (risk).
        
        For equal-weighted portfolio:
            σ² = w^T Σ w
        where wᵢ = 1/k for selected assets, 0 otherwise
        
        Returns standard deviation (sqrt of variance).
        """
        # Equal weights for selected assets
        selected_indices = [i for i, x in enumerate(solution) if x == 1]
        k = len(selected_indices)
        
        if k == 0:
            return 0.0
        
        # Weight vector
        weights = np.zeros(self.num_assets)
        weights[selected_indices] = 1.0 / k
        
        # Portfolio variance: w^T Σ w
        portfolio_variance = weights @ self.covariance_matrix @ weights
        
        # Return standard deviation (volatility)
        return np.sqrt(portfolio_variance)
    
    def to_graph(self) -> nx.Graph:
        """
        Convert portfolio to correlation graph.
        
        Creates a graph where:
        - Nodes represent assets
        - Edge weights represent correlations
        - Node attributes include expected return and volatility
        
        This representation is useful for:
        - Visualizing asset relationships
        - Identifying diversification opportunities
        - Cluster analysis
        
        Returns:
            NetworkX graph with correlation edges
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = PortfolioProblem(num_assets=10, num_selected=3)
            >>> problem.generate()
            >>> graph = problem.to_graph()
            >>> 
            >>> # Analyze correlations
            >>> for u, v, data in graph.edges(data=True):
            ...     if data['weight'] > 0.7:  # High correlation
            ...         print(f"Assets {u} and {v} are highly correlated")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Extract correlation matrix from covariance
        volatilities = np.sqrt(np.diag(self.covariance_matrix))
        correlation_matrix = (
            self.covariance_matrix / np.outer(volatilities, volatilities)
        )
        
        # Create complete graph
        graph = nx.complete_graph(self.num_assets)
        
        # Add edge weights (correlations)
        for i in range(self.num_assets):
            for j in range(i + 1, self.num_assets):
                correlation = correlation_matrix[i, j]
                graph[i][j]['weight'] = correlation
        
        # Add node attributes
        for i in range(self.num_assets):
            graph.nodes[i]['return'] = self.expected_returns[i]
            graph.nodes[i]['volatility'] = volatilities[i]
            graph.nodes[i]['sharpe'] = (
                (self.expected_returns[i] - self.risk_free_rate) / volatilities[i]
            )
        
        return graph
    
    def to_qubo(self) -> np.ndarray:
        """
        Convert portfolio optimization to QUBO form.
        
        QUBO Formulation:
        ----------------
        Objective: Maximize Sharpe ratio (or minimize -Sharpe)
        
        For equal-weighted portfolios:
            Sharpe ≈ (Σᵢ xᵢ μᵢ - k·r_f) / √(Σᵢⱼ xᵢ xⱼ Σᵢⱼ)
        
        This is complex to encode exactly. We use approximation:
        
        1. Maximize returns: max Σᵢ xᵢ μᵢ
        2. Minimize risk: min Σᵢⱼ xᵢ xⱼ Σᵢⱼ
        3. Enforce cardinality: (Σᵢ xᵢ - k)² = 0
        
        Combined QUBO:
            Q = -λ₁·R + λ₂·Σ + λ₃·(cardinality penalty)
        
        where:
        - R: Return matrix (diagonal with expected returns)
        - Σ: Covariance matrix (risk)
        - λ₁, λ₂, λ₃: Weighting coefficients
        
        The cardinality constraint penalty:
            P·(Σᵢ xᵢ - k)² = P·(Σᵢ xᵢ² - 2k·Σᵢ xᵢ + k²)
                           = P·(n - 2k·Σᵢ xᵢ + k²)  (since xᵢ² = xᵢ)
        
        This expands to:
        - Diagonal: P·(1 - 2k)
        - Off-diagonal: 2P
        
        Returns:
            QUBO matrix (n × n)
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = PortfolioProblem(num_assets=10, num_selected=3)
            >>> problem.generate()
            >>> qubo = problem.to_qubo()
            >>> 
            >>> # Use with quantum solver (QAOA, quantum annealing)
            >>> # solution = quantum_solver.solve(qubo)
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        n = self.num_assets
        k = self.num_selected
        
        # QUBO matrix
        qubo = np.zeros((n, n), dtype=np.float64)
        
        # Weighting coefficients (tunable hyperparameters)
        lambda_return = 1.0    # Weight for return maximization
        lambda_risk = 0.5      # Weight for risk minimization
        lambda_cardinality = 10.0  # Penalty for cardinality constraint
        
        # 1. Return term: -λ₁·μᵢ on diagonal (negative for maximization)
        for i in range(n):
            qubo[i, i] -= lambda_return * self.expected_returns[i]
        
        # 2. Risk term: +λ₂·Σᵢⱼ in upper triangle
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    qubo[i, j] += lambda_risk * self.covariance_matrix[i, j]
                else:
                    qubo[i, j] += 2.0 * lambda_risk * self.covariance_matrix[i, j]
        
        # 3. Cardinality constraint penalty
        # Diagonal: P·(1 - 2k)
        # Off-diagonal: 2P
        for i in range(n):
            qubo[i, i] += lambda_cardinality * (1 - 2 * k)
        
        for i in range(n):
            for j in range(i + 1, n):
                qubo[i, j] += 2.0 * lambda_cardinality
        
        # Ensure upper triangular
        qubo = np.triu(qubo)
        
        return qubo
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get portfolio problem characteristics.
        
        Returns:
            Dictionary containing:
            - problem_type: 'portfolio'
            - problem_size: Number of assets
            - complexity_class: 'NP-hard'
            - num_selected: Cardinality constraint
            - avg_return: Mean expected return
            - avg_volatility: Mean asset volatility
            - avg_correlation: Average pairwise correlation
            - diversification_ratio: Measure of diversification benefit
            - risk_free_rate: Baseline return rate
            - sparsity: 1 - (num_selected / num_assets)
            - estimated_classical_time: Time for exact solution
            - estimated_quantum_advantage: Potential quantum speedup
        
        Raises:
            ValueError: If problem not generated
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Extract volatilities and correlations
        volatilities = np.sqrt(np.diag(self.covariance_matrix))
        correlation_matrix = (
            self.covariance_matrix / np.outer(volatilities, volatilities)
        )
        
        # Average correlation (exclude diagonal)
        upper_triangle = correlation_matrix[np.triu_indices(self.num_assets, k=1)]
        avg_correlation = np.mean(upper_triangle)
        
        # Diversification ratio (higher is better)
        # Ratio of weighted average volatility to portfolio volatility
        # Perfect diversification = 1.0, no diversification < 1.0
        equal_weights = np.ones(self.num_assets) / self.num_assets
        portfolio_vol = np.sqrt(equal_weights @ self.covariance_matrix @ equal_weights)
        avg_vol = np.mean(volatilities)
        diversification_ratio = avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Estimated solution time
        # Brute force: O(C(n,k)) = O(n^k / k!)
        from scipy.special import comb
        num_combinations = comb(self.num_assets, self.num_selected, exact=True)
        classical_time_estimate = num_combinations / 1e6  # Rough estimate
        
        # Quantum advantage estimate
        # Portfolio optimization shows promise for quantum
        if self.num_assets <= 20:
            quantum_advantage = 1.0  # Small problems: classical OK
        elif 20 < self.num_assets <= 100:
            quantum_advantage = 2.5  # Medium: good quantum candidate
        else:
            quantum_advantage = 1.8  # Large: quantum helps but QUBO size grows
        
        return {
            'problem_type': self.problem_type,
            'problem_size': self.problem_size,
            'complexity_class': self.complexity_class,
            'num_selected': self.num_selected,
            'avg_return': float(np.mean(self.expected_returns)),
            'avg_volatility': float(np.mean(volatilities)),
            'avg_correlation': float(avg_correlation),
            'diversification_ratio': float(diversification_ratio),
            'risk_free_rate': self.risk_free_rate,
            'sparsity': 1.0 - (self.num_selected / self.num_assets),
            'symmetry': True,  # Covariance matrix is symmetric
            'structure': 'complete',  # All asset pairs have correlation
            'estimated_classical_time': classical_time_estimate,
            'estimated_quantum_advantage': quantum_advantage,
        }
    
    def get_optimal_solution_brute_force(self) -> Tuple[List[int], float]:
        """
        Find optimal portfolio via exhaustive search.
        
        WARNING: Combinatorial complexity C(n, k). Feasible only when
        C(n, k) < 10 million (e.g., n=20, k=5 gives ~15k combinations).
        
        Returns:
            Tuple of (optimal_portfolio, optimal_sharpe_ratio)
        
        Raises:
            ValueError: If problem not generated or too large
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        from scipy.special import comb
        num_combinations = comb(self.num_assets, self.num_selected, exact=True)
        
        if num_combinations > 10_000_000:
            raise ValueError(
                f"Brute force infeasible: {num_combinations:,} combinations "
                f"(n={self.num_assets}, k={self.num_selected})"
            )
        
        best_portfolio = None
        best_sharpe = -np.inf
        
        # Enumerate all combinations
        for selected_assets in combinations(range(self.num_assets), self.num_selected):
            # Convert to binary vector
            portfolio = [0] * self.num_assets
            for idx in selected_assets:
                portfolio[idx] = 1
            
            # Evaluate
            cost = self.calculate_cost(portfolio)
            sharpe = -cost
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_portfolio = portfolio
        
        return best_portfolio, best_sharpe
    
    def get_random_solution(self, seed: Optional[int] = None) -> List[int]:
        """Generate random valid portfolio (k assets selected randomly)."""
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Random selection of k assets
        selected = np.random.choice(
            self.num_assets, size=self.num_selected, replace=False
        )
        
        portfolio = [0] * self.num_assets
        for idx in selected:
            portfolio[idx] = 1
        
        return portfolio
    
    def visualize_efficient_frontier(
        self,
        num_portfolios: int = 1000,
        highlight_solution: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        show: bool = True,
    ) -> None:
        """
        Visualize the efficient frontier (risk-return tradeoff).
        
        The efficient frontier shows the best possible risk-return combinations.
        Points above the frontier are achievable, points below are not optimal.
        
        Args:
            num_portfolios: Number of random portfolios to plot
            highlight_solution: Specific portfolio to highlight (optional)
            save_path: Path to save figure (optional)
            figsize: Figure size
            show: Whether to display
        
        Example:
            >>> problem = PortfolioProblem(num_assets=20, num_selected=5)
            >>> problem.generate()
            >>> 
            >>> # Find optimal
            >>> optimal, _ = problem.get_optimal_solution_brute_force()
            >>> 
            >>> # Visualize with optimal highlighted
            >>> problem.visualize_efficient_frontier(
            ...     num_portfolios=500,
            ...     highlight_solution=optimal,
            ...     save_path="efficient_frontier.png"
            ... )
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Generate random portfolios
        risks = []
        returns = []
        sharpes = []
        
        for _ in range(num_portfolios):
            portfolio = self.get_random_solution()
            
            # Calculate metrics
            selected = [i for i, x in enumerate(portfolio) if x == 1]
            port_return = np.mean(self.expected_returns[selected])
            port_risk = self._calculate_portfolio_risk(portfolio)
            sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
            
            returns.append(port_return)
            risks.append(port_risk)
            sharpes.append(sharpe)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot colored by Sharpe ratio
        scatter = ax.scatter(
            risks, returns,
            c=sharpes, cmap='RdYlGn',
            alpha=0.6, s=50, edgecolors='black', linewidths=0.5
        )
        
        # Highlight specific solution
        if highlight_solution is not None:
            selected = [i for i, x in enumerate(highlight_solution) if x == 1]
            port_return = np.mean(self.expected_returns[selected])
            port_risk = self._calculate_portfolio_risk(highlight_solution)
            sharpe = (port_return - self.risk_free_rate) / port_risk
            
            ax.scatter(
                [port_risk], [port_return],
                c='red', s=300, marker='*',
                edgecolors='black', linewidths=2,
                label=f'Selected (Sharpe={sharpe:.3f})', zorder=5
            )
        
        # Color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('Portfolio Risk (Volatility)', fontsize=14)
        ax.set_ylabel('Expected Return', fontsize=14)
        ax.set_title(
            f'Efficient Frontier ({self.num_assets} assets, select {self.num_selected})',
            fontsize=16, fontweight='bold'
        )
        
        # Add risk-free rate line
        ax.axhline(
            y=self.risk_free_rate, color='blue', linestyle='--',
            alpha=0.5, label=f'Risk-free rate ({self.risk_free_rate:.2%})'
        )
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
