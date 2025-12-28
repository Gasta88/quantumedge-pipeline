"""
Abstract base class for optimization problems in QuantumEdge Pipeline.

This module defines the interface that all optimization problems must implement,
enabling consistent handling across classical, quantum, and hybrid solvers.

Problem Representations
-----------------------
Different solvers require different problem representations:

1. **Graph Representation (NetworkX)**:
   - Used by classical graph algorithms (e.g., Goemans-Williamson for MaxCut)
   - Provides rich graph manipulation and analysis tools
   - Natural representation for problems with pairwise relationships

2. **QUBO Representation (Quadratic Unconstrained Binary Optimization)**:
   - Required for quantum annealing and gate-based quantum algorithms
   - Mathematical formulation: minimize x^T Q x, where x ∈ {0,1}^n
   - Q is an upper-triangular matrix encoding problem structure
   - Standard input format for quantum solvers (D-Wave, QAOA, VQE)

Why Multiple Representations?
------------------------------
- Classical solvers work best with structured graph representations
- Quantum solvers require QUBO or Ising model formulations
- Routing decisions depend on problem characteristics visible in different forms
- Conversion between representations enables hybrid approaches

Example Usage
-------------
```python
from src.problems.maxcut import MaxCutProblem

# Create problem instance
problem = MaxCutProblem(num_nodes=20, edge_probability=0.3)
problem.generate()

# Validate a candidate solution
solution = [0, 1, 0, 1, ...]  # Binary assignment
is_valid = problem.validate_solution(solution)

# Evaluate solution quality
cost = problem.calculate_cost(solution)

# Get different representations
graph = problem.to_graph()  # NetworkX graph
qubo = problem.to_qubo()    # QUBO matrix for quantum solver

# Get problem characteristics for routing
metadata = problem.get_metadata()
print(f"Complexity: {problem.complexity_class}")
print(f"Sparsity: {metadata['sparsity']}")
```
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import networkx as nx


class ProblemBase(ABC):
    """
    Abstract base class for optimization problems.
    
    All optimization problems (MaxCut, TSP, Portfolio, etc.) must inherit
    from this class and implement the required abstract methods. This ensures
    consistent interfaces for problem generation, solution validation,
    cost evaluation, and representation conversion.
    
    The class provides a unified interface that allows the quantum-classical
    router to analyze problem characteristics and make informed routing
    decisions without knowing specific problem details.
    
    Attributes:
        problem_type (str): Type identifier ('maxcut', 'tsp', 'portfolio')
        problem_size (int): Problem dimension (nodes, cities, assets)
        complexity_class (str): Computational complexity ('P', 'NP', 'NP-hard')
    
    Subclass Implementation Requirements:
        - Implement all @abstractmethod methods
        - Set problem_type, problem_size, complexity_class in __init__
        - Ensure generate() creates valid, solvable problem instances
        - Make to_qubo() return upper-triangular matrix (standard QUBO form)
    """
    
    def __init__(self):
        """
        Initialize base problem instance.
        
        Subclasses should call super().__init__() and then set:
        - self._problem_type
        - self._problem_size
        - self._complexity_class
        """
        self._problem_type: str = "unknown"
        self._problem_size: int = 0
        self._complexity_class: str = "unknown"
        self._generated: bool = False
    
    @property
    def problem_type(self) -> str:
        """
        Get the problem type identifier.
        
        Returns:
            Problem type string ('maxcut', 'tsp', 'portfolio', etc.)
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=10)
            >>> print(problem.problem_type)  # 'maxcut'
        """
        return self._problem_type
    
    @property
    def problem_size(self) -> int:
        """
        Get the problem size (dimensionality).
        
        Returns:
            Problem size as integer:
            - MaxCut: number of nodes
            - TSP: number of cities
            - Portfolio: number of assets
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=20)
            >>> print(problem.problem_size)  # 20
        """
        return self._problem_size
    
    @property
    def complexity_class(self) -> str:
        """
        Get the computational complexity class.
        
        Complexity classes indicate worst-case time complexity:
        - 'P': Polynomial time (efficient classical algorithms exist)
        - 'NP': Non-deterministic polynomial (solution verification is polynomial)
        - 'NP-hard': At least as hard as hardest NP problems (no known polynomial solution)
        - 'NP-complete': Both NP and NP-hard
        
        This information helps the router decide between classical and quantum:
        - P problems: Classical solvers preferred
        - NP-hard problems: Potential quantum advantage for certain instances
        
        Returns:
            Complexity class string ('P', 'NP', 'NP-hard', 'NP-complete')
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=10)
            >>> print(problem.complexity_class)  # 'NP-hard'
        """
        return self._complexity_class
    
    @property
    def is_generated(self) -> bool:
        """
        Check if problem instance has been generated.
        
        Returns:
            True if generate() has been called successfully
        """
        return self._generated
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def generate(self, seed: Optional[int] = None, **kwargs) -> None:
        """
        Generate a random problem instance.
        
        Creates a concrete problem instance with random structure based on
        problem-specific parameters. This method should initialize all
        internal data structures needed for solution evaluation and
        representation conversion.
        
        After successful generation, sets self._generated = True.
        
        Args:
            seed: Random seed for reproducibility (optional)
            **kwargs: Problem-specific parameters, e.g.:
                     - MaxCut: edge_probability, weight_range
                     - TSP: distance_range, euclidean
                     - Portfolio: correlation_strength, return_range
        
        Raises:
            ValueError: If parameters are invalid
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=20)
            >>> problem.generate(seed=42, edge_probability=0.3)
            >>> assert problem.is_generated
        
        Implementation Notes:
            - Use provided seed for numpy/random generators
            - Validate all kwargs before generation
            - Ensure generated instance is solvable
            - Set self._generated = True at the end
        """
        pass
    
    @abstractmethod
    def validate_solution(self, solution: List[Any]) -> bool:
        """
        Validate that a solution satisfies problem constraints.
        
        Checks if a candidate solution is feasible according to problem-specific
        constraints. Does NOT evaluate solution quality (use calculate_cost for that).
        
        Validation checks depend on problem type:
        - MaxCut: Binary assignment, correct length
        - TSP: Valid tour (all cities visited once), correct length
        - Portfolio: Valid allocation, sum to 100%, non-negative
        
        Args:
            solution: Candidate solution in problem-specific format:
                     - MaxCut: List[int] binary assignments [0,1,0,1,...]
                     - TSP: List[int] city ordering [0,3,1,2,4,...]
                     - Portfolio: List[float] weights [0.2, 0.3, 0.5, ...]
        
        Returns:
            True if solution is valid, False otherwise
        
        Raises:
            ValueError: If problem not generated yet
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=4)
            >>> problem.generate()
            >>> solution = [0, 1, 0, 1]
            >>> assert problem.validate_solution(solution)
            >>> invalid = [0, 2, 0, 1]  # Invalid: not binary
            >>> assert not problem.validate_solution(invalid)
        
        Implementation Notes:
            - Check self._generated first
            - Verify solution format and length
            - Check problem-specific constraints
            - Return False (don't raise) for invalid solutions
        """
        pass
    
    @abstractmethod
    def calculate_cost(self, solution: List[Any]) -> float:
        """
        Calculate the objective function value for a solution.
        
        Evaluates solution quality according to the problem's objective function.
        Convention: Lower cost = better solution (minimization). For maximization
        problems (like MaxCut), return negative of objective value.
        
        Cost calculation:
        - MaxCut: -(sum of cut edges weights) [negative because we maximize cut]
        - TSP: total tour distance [minimize]
        - Portfolio: -expected_return + risk_penalty [minimize risk, maximize return]
        
        Args:
            solution: Valid solution in problem-specific format
        
        Returns:
            Objective function value (lower is better)
        
        Raises:
            ValueError: If problem not generated or solution invalid
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=4)
            >>> problem.generate()
            >>> solution = [0, 1, 0, 1]
            >>> cost = problem.calculate_cost(solution)
            >>> print(f"Cut value: {-cost}")  # Negative because MaxCut maximizes
        
        Implementation Notes:
            - Call validate_solution() first
            - Use efficient calculation (avoid redundant loops)
            - Return float (not int) for precision
            - Follow minimization convention
        """
        pass
    
    @abstractmethod
    def to_graph(self) -> nx.Graph:
        """
        Convert problem to NetworkX graph representation.
        
        Creates a NetworkX graph that captures the problem structure. This
        representation is used by:
        - Classical graph algorithms (e.g., minimum spanning tree, max flow)
        - Graph analysis tools (centrality, clustering coefficient)
        - Visualization and debugging
        - Problem characteristic analysis for routing decisions
        
        Graph structure by problem type:
        - MaxCut: Undirected weighted graph, nodes=variables, edges=interactions
        - TSP: Complete graph, nodes=cities, edge weights=distances
        - Portfolio: Nodes=assets, edges=correlations (if applicable)
        
        Node and Edge Attributes:
            - Nodes can have attributes like 'weight', 'label', 'position'
            - Edges should have 'weight' attribute
            - Use consistent naming for cross-problem compatibility
        
        Returns:
            NetworkX Graph (or DiGraph for directed problems)
        
        Raises:
            ValueError: If problem not generated yet
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=5)
            >>> problem.generate()
            >>> graph = problem.to_graph()
            >>> print(f"Nodes: {graph.number_of_nodes()}")
            >>> print(f"Edges: {graph.number_of_edges()}")
            >>> print(f"Density: {nx.density(graph)}")
        
        Implementation Notes:
            - Check self._generated first
            - Create appropriate graph type (Graph/DiGraph)
            - Add all nodes and edges with attributes
            - Return lightweight graph (avoid copying large data)
        """
        pass
    
    @abstractmethod
    def to_qubo(self) -> np.ndarray:
        """
        Convert problem to QUBO (Quadratic Unconstrained Binary Optimization) form.
        
        QUBO is the standard input format for quantum optimization algorithms:
        - Quantum annealing (D-Wave)
        - QAOA (Quantum Approximate Optimization Algorithm)
        - VQE (Variational Quantum Eigensolver)
        
        What is QUBO?
        -------------
        QUBO minimizes: f(x) = x^T Q x = Σᵢⱼ Qᵢⱼ xᵢ xⱼ
        where:
        - x ∈ {0,1}ⁿ is a binary vector (decision variables)
        - Q is an n×n upper-triangular matrix (QUBO matrix)
        - Qᵢᵢ (diagonal): linear terms (bias on variable i)
        - Qᵢⱼ (i<j): quadratic terms (interaction between variables i and j)
        
        Why Upper-Triangular?
        - Since xᵢxⱼ = xⱼxᵢ for binary variables
        - Convention: store interaction in upper triangle only
        - Lower triangle should be zeros (or ignored)
        
        Conversion Examples:
        - MaxCut: Qᵢⱼ = -wᵢⱼ (negative edge weights)
        - TSP: Complex encoding with constraints as penalties
        - Portfolio: Quadratic form of risk-return tradeoff
        
        Returns:
            Upper-triangular QUBO matrix as numpy array (shape: n×n)
            where n = problem_size
        
        Raises:
            ValueError: If problem not generated or QUBO conversion infeasible
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=4)
            >>> problem.generate()
            >>> qubo = problem.to_qubo()
            >>> print(f"QUBO shape: {qubo.shape}")  # (4, 4)
            >>> # Diagonal contains linear terms
            >>> print(f"Linear terms: {np.diag(qubo)}")
            >>> # Upper triangle contains interactions
            >>> print(f"Is upper triangular: {np.allclose(qubo, np.triu(qubo))}")
        
        Implementation Notes:
            - Check self._generated first
            - Return upper-triangular matrix (zero lower triangle)
            - Use float64 for numerical precision
            - Include penalty terms for constraint violations (if needed)
            - Verify QUBO correctness: evaluate solutions and compare with calculate_cost()
        
        QUBO to Ising Conversion (for reference):
        -----------------------------------------
        Quantum computers often use Ising model: H = Σᵢⱼ Jᵢⱼ sᵢsⱼ + Σᵢ hᵢsᵢ
        where s ∈ {-1, +1}
        
        Convert using: xᵢ = (1 - sᵢ)/2
        This is typically handled by quantum solver libraries.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return problem characteristics for routing and analysis.
        
        Provides metadata that helps the quantum-classical router make
        informed decisions about solver selection. Includes structural
        properties, complexity estimates, and quantum advantage indicators.
        
        Required Metadata Keys:
            - 'problem_type': str (e.g., 'maxcut')
            - 'problem_size': int (e.g., 50)
            - 'complexity_class': str (e.g., 'NP-hard')
            - 'sparsity': float (0-1, edge density for graphs)
            - 'symmetry': bool (is problem symmetric?)
            - 'structure': str ('random', 'clustered', 'sparse', 'dense')
            - 'estimated_classical_time': float (seconds)
            - 'estimated_quantum_advantage': float (potential speedup ratio)
        
        Optional Metadata (problem-specific):
            - 'num_edges': int (for graph problems)
            - 'diameter': int (graph diameter)
            - 'clustering_coefficient': float
            - 'constraint_count': int (for constrained problems)
            - 'qubo_size': int (QUBO matrix dimension)
        
        Returns:
            Dictionary of problem characteristics
        
        Raises:
            ValueError: If problem not generated yet
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=50)
            >>> problem.generate(edge_probability=0.2)
            >>> metadata = problem.get_metadata()
            >>> print(f"Sparsity: {metadata['sparsity']:.2f}")
            >>> print(f"Structure: {metadata['structure']}")
            >>> print(f"Estimated quantum advantage: {metadata['estimated_quantum_advantage']:.2f}x")
            >>> 
            >>> # Router uses this for decisions
            >>> if metadata['sparsity'] < 0.3 and metadata['problem_size'] > 30:
            ...     route_to = "quantum"  # Sparse, medium-sized -> quantum advantage
        
        Implementation Notes:
            - Check self._generated first
            - Calculate sparsity from graph representation
            - Estimate classical time based on problem size and structure
            - Estimate quantum advantage based on literature/benchmarks
            - Cache expensive computations
            - Return consistent structure across problem types
        """
        pass
    
    # =========================================================================
    # Optional Helper Methods (can be overridden)
    # =========================================================================
    
    def get_optimal_solution(self) -> Optional[List[Any]]:
        """
        Return the optimal solution if known (for small benchmark instances).
        
        For benchmarking and testing, some problems may have precomputed
        optimal solutions. This is typically only feasible for small instances.
        
        Returns:
            Optimal solution if known, None otherwise
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=4)
            >>> problem.generate(seed=42)  # Known instance
            >>> optimal = problem.get_optimal_solution()
            >>> if optimal:
            ...     print(f"Optimal cost: {problem.calculate_cost(optimal)}")
        """
        return None
    
    def get_random_solution(self, seed: Optional[int] = None) -> List[Any]:
        """
        Generate a random valid solution.
        
        Useful for:
        - Initializing optimization algorithms
        - Benchmarking solver performance
        - Testing solution validation
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Random valid solution
        
        Raises:
            ValueError: If problem not generated yet
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=10)
            >>> problem.generate()
            >>> random_sol = problem.get_random_solution(seed=42)
            >>> assert problem.validate_solution(random_sol)
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Default implementation for binary problems
        # Override for problem-specific logic
        if seed is not None:
            np.random.seed(seed)
        return list(np.random.randint(0, 2, self.problem_size))
    
    def __repr__(self) -> str:
        """String representation of the problem."""
        status = "generated" if self._generated else "not generated"
        return (
            f"{self.__class__.__name__}("
            f"type='{self.problem_type}', "
            f"size={self.problem_size}, "
            f"complexity='{self.complexity_class}', "
            f"status='{status}')"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"{self.problem_type.upper()} Problem "
            f"(size={self.problem_size}, {self.complexity_class})"
        )
