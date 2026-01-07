"""
Traveling Salesman Problem (TSP) Implementation.

The Traveling Salesman Problem is one of the most famous combinatorial
optimization problems in computer science and operations research.

Problem Definition:
    Given a set of cities and distances between each pair of cities, find
    the shortest possible route that visits each city exactly once and
    returns to the origin city.
    
    Mathematically:
        TSP(cities) = min_{π∈Π} Σ_{i=0}^{n-1} d(city[π[i]], city[π[i+1]])
    
    where:
    - π is a permutation of cities (a tour)
    - Π is the set of all possible permutations
    - d(a, b) is the distance between cities a and b
    - city[π[n]] = city[π[0]] (return to start)

Complexity:
    - NP-hard: No known polynomial-time algorithm for optimal solution
    - Brute force: O(n!) - infeasible for n > 15
    - Best approximation: Christofides algorithm (1.5-approximation for metric TSP)
    - Practical solvers: Branch-and-bound, dynamic programming (Held-Karp O(n²·2ⁿ))

Why TSP is Hard:
    1. Combinatorial explosion: n! possible tours for n cities
       - 10 cities: 3,628,800 tours
       - 20 cities: 2.4 × 10¹⁸ tours (infeasible to enumerate)
    
    2. No local structure: Small changes can drastically affect total distance
       (unlike problems with independent subproblems)
    
    3. Global constraints: Must visit all cities exactly once
       (not easily decomposable)
    
    4. No efficient heuristics guarantee optimality
       (best we can do is approximations)

QUBO Conversion Complexity:
    Converting TSP to QUBO is significantly more complex than MaxCut:
    
    1. Variable encoding:
       - Need O(n²) binary variables: x_{i,t} ∈ {0,1}
       - x_{i,t} = 1 means "visit city i at time step t"
    
    2. Constraint penalties:
       - Each city visited exactly once: Σ_t x_{i,t} = 1 for all i
       - Each time step visits exactly one city: Σ_i x_{i,t} = 1 for all t
       - These become quadratic penalty terms in QUBO
    
    3. Distance objective:
       - Σ_{i,j,t} d_{ij} · x_{i,t} · x_{j,t+1}
       - Requires O(n³) terms
    
    4. Result: QUBO matrix is O(n²) × O(n²) = O(n⁴) in size!
       - Much larger than original problem
       - Not practical for large n
    
    This implementation uses a SIMPLIFIED QUBO for demonstration purposes.
    Real-world quantum TSP solvers use specialized encodings (e.g., QAOA with
    problem-specific ansatzes, or quantum annealing with embedding techniques).

Applications:
    - Logistics and delivery routing
    - Circuit board drilling (optimize drill path)
    - DNA sequencing (fragment assembly)
    - Telescope scheduling (minimize slewing time)
    - Manufacturing (robot arm movement optimization)

Example Usage:
    >>> from src.problems.tsp import TSPProblem
    >>> 
    >>> # Create TSP with 10 cities
    >>> problem = TSPProblem(num_cities=10)
    >>> 
    >>> # Generate random cities in 2D plane
    >>> problem.generate(seed=42, coordinate_range=(0, 100))
    >>> 
    >>> # Get a random tour
    >>> tour = problem.get_random_solution(seed=123)
    >>> 
    >>> # Validate and evaluate
    >>> assert problem.validate_solution(tour)
    >>> cost = problem.calculate_cost(tour)
    >>> print(f"Tour length: {cost:.2f}")
    >>> 
    >>> # Visualize the tour
    >>> problem.visualize(tour, save_path="tsp_tour.png")
    >>> 
    >>> # Get problem metadata
    >>> metadata = problem.get_metadata()
    >>> print(f"Problem type: {metadata['problem_type']}")
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import networkx as nx
from itertools import permutations
import matplotlib.pyplot as plt

from src.problems.problem_base import ProblemBase


class TSPProblem(ProblemBase):
    """
    Traveling Salesman Problem (TSP) implementation.
    
    Represents a TSP instance where cities have 2D coordinates and distances
    are Euclidean. The goal is to find the shortest tour visiting all cities
    exactly once and returning to the start.
    
    This implementation provides:
    - Random city generation in 2D plane
    - Euclidean distance calculation
    - Tour validation and cost evaluation
    - Simplified QUBO conversion (for demonstration)
    - Visualization with matplotlib
    - Brute force solver for small instances
    
    Attributes:
        num_cities (int): Number of cities
        coordinates (np.ndarray): City coordinates (n × 2 array)
        distance_matrix (np.ndarray): Pairwise distances (n × n matrix)
    
    Note:
        QUBO conversion is SIMPLIFIED for demonstration. Real TSP quantum
        solvers require specialized techniques beyond standard QUBO formulation.
    
    Example:
        >>> # Small instance - can find optimal
        >>> problem = TSPProblem(num_cities=8)
        >>> problem.generate(seed=42)
        >>> 
        >>> # Find optimal tour (feasible for n <= 10)
        >>> optimal_tour, optimal_distance = problem.get_optimal_solution_brute_force()
        >>> print(f"Optimal tour length: {optimal_distance:.2f}")
        >>> 
        >>> # Visualize
        >>> problem.visualize(optimal_tour)
    """
    
    def __init__(self, num_cities: int):
        """
        Initialize TSP problem instance.
        
        Args:
            num_cities: Number of cities (must be >= 3 for meaningful TSP)
        
        Raises:
            ValueError: If num_cities < 3
        
        Example:
            >>> problem = TSPProblem(num_cities=15)
            >>> print(problem)  # TSP Problem (size=15, NP-hard)
        """
        super().__init__()
        
        if num_cities < 3:
            raise ValueError("num_cities must be at least 3 for TSP")
        
        self._problem_type = "tsp"
        self._problem_size = num_cities
        self._complexity_class = "NP-hard"
        
        self.num_cities = num_cities
        self.coordinates: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
    
    def generate(
        self,
        seed: Optional[int] = None,
        coordinate_range: Tuple[float, float] = (0.0, 100.0),
        **kwargs
    ) -> None:
        """
        Generate random cities with 2D coordinates.
        
        Cities are placed randomly in a 2D plane with coordinates sampled
        uniformly from the specified range. Distances are computed as
        Euclidean distances between city coordinates.
        
        Args:
            seed: Random seed for reproducibility
            coordinate_range: (min, max) for x and y coordinates
            **kwargs: Additional arguments (ignored)
        
        Raises:
            ValueError: If coordinate_range is invalid
        
        Example:
            >>> problem = TSPProblem(num_cities=20)
            >>> 
            >>> # Cities in [0, 100] × [0, 100] square
            >>> problem.generate(seed=42, coordinate_range=(0, 100))
            >>> 
            >>> # Cities in larger area
            >>> problem.generate(seed=123, coordinate_range=(0, 1000))
        """
        if coordinate_range[0] >= coordinate_range[1]:
            raise ValueError("coordinate_range must be (min, max) with min < max")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random 2D coordinates for cities
        min_coord, max_coord = coordinate_range
        self.coordinates = np.random.uniform(
            min_coord, max_coord,
            size=(self.num_cities, 2)
        )
        
        # Compute pairwise Euclidean distances
        self.distance_matrix = self._compute_distance_matrix()
        
        self._generated = True
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Compute Euclidean distance matrix between all city pairs.
        
        Uses vectorized numpy operations for efficiency.
        
        Returns:
            Symmetric distance matrix (n × n) where d[i][j] is distance
            from city i to city j
        """
        n = self.num_cities
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)
                dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                distances[i, j] = dist
                distances[j, i] = dist  # Symmetric
        
        return distances
    
    def validate_solution(self, solution: List[int]) -> bool:
        """
        Validate a TSP tour solution.
        
        A valid TSP tour must:
        1. Visit all cities exactly once (permutation of [0, 1, ..., n-1])
        2. Have correct length (n cities)
        3. Contain valid city indices
        
        Args:
            solution: Tour as list of city indices [c0, c1, c2, ..., cn-1]
                     Tour implicitly returns to c0 (circular)
        
        Returns:
            True if solution is a valid tour, False otherwise
        
        Example:
            >>> problem = TSPProblem(num_cities=5)
            >>> problem.generate()
            >>> 
            >>> valid = [0, 1, 2, 3, 4]  # Valid tour
            >>> assert problem.validate_solution(valid)
            >>> 
            >>> invalid = [0, 1, 2, 3]  # Too short
            >>> assert not problem.validate_solution(invalid)
            >>> 
            >>> invalid = [0, 1, 2, 2, 3]  # City 2 visited twice
            >>> assert not problem.validate_solution(invalid)
            >>> 
            >>> invalid = [0, 1, 2, 3, 5]  # Invalid city index 5
            >>> assert not problem.validate_solution(invalid)
        """
        if not self._generated:
            return False
        
        # Check length
        if len(solution) != self.num_cities:
            return False
        
        # Check all values are valid city indices
        if not all(0 <= city < self.num_cities for city in solution):
            return False
        
        # Check all cities visited exactly once (is a permutation)
        if len(set(solution)) != self.num_cities:
            return False
        
        return True
    
    def calculate_cost(self, solution: List[int]) -> float:
        """
        Calculate total tour distance.
        
        Computes the sum of distances between consecutive cities in the tour,
        including the distance from the last city back to the first city
        (completing the cycle).
        
        Mathematical formulation:
            cost = Σ_{i=0}^{n-1} d(city[i], city[(i+1) mod n])
        
        Args:
            solution: Tour as list of city indices
        
        Returns:
            Total tour distance (Euclidean)
        
        Raises:
            ValueError: If solution is invalid or problem not generated
        
        Example:
            >>> problem = TSPProblem(num_cities=6)
            >>> problem.generate(seed=42)
            >>> 
            >>> # Sequential tour
            >>> tour1 = [0, 1, 2, 3, 4, 5]
            >>> cost1 = problem.calculate_cost(tour1)
            >>> 
            >>> # Different tour
            >>> tour2 = [0, 2, 4, 1, 3, 5]
            >>> cost2 = problem.calculate_cost(tour2)
            >>> 
            >>> print(f"Tour 1 distance: {cost1:.2f}")
            >>> print(f"Tour 2 distance: {cost2:.2f}")
        """
        if not self.validate_solution(solution):
            raise ValueError("Invalid tour solution")
        
        total_distance = 0.0
        
        # Sum distances between consecutive cities
        for i in range(self.num_cities):
            from_city = solution[i]
            to_city = solution[(i + 1) % self.num_cities]  # Wrap around at end
            total_distance += self.distance_matrix[from_city, to_city]
        
        return total_distance
    
    def to_graph(self) -> nx.Graph:
        """
        Convert TSP to complete graph with distance weights.
        
        Creates a complete graph where:
        - Nodes represent cities (labeled 0 to n-1)
        - Edges connect all pairs of cities
        - Edge weights are Euclidean distances
        - Node attributes include 'pos' (x, y coordinates)
        
        Returns:
            NetworkX complete graph with distance weights
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = TSPProblem(num_cities=10)
            >>> problem.generate()
            >>> graph = problem.to_graph()
            >>> 
            >>> print(f"Nodes: {graph.number_of_nodes()}")  # 10
            >>> print(f"Edges: {graph.number_of_edges()}")  # 45 (complete graph)
            >>> 
            >>> # Access city coordinates
            >>> for node, data in graph.nodes(data=True):
            ...     x, y = data['pos']
            ...     print(f"City {node}: ({x:.1f}, {y:.1f})")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Create complete graph
        graph = nx.complete_graph(self.num_cities)
        
        # Add edge weights (distances)
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                distance = self.distance_matrix[i, j]
                graph[i][j]['weight'] = distance
        
        # Add node positions
        for i in range(self.num_cities):
            graph.nodes[i]['pos'] = tuple(self.coordinates[i])
        
        return graph
    
    def to_qubo(self) -> np.ndarray:
        """
        Convert TSP to QUBO form (SIMPLIFIED VERSION).
        
        WARNING: This is a SIMPLIFIED demonstration of TSP QUBO conversion.
        Real TSP quantum solvers use more sophisticated encodings.
        
        Full TSP QUBO Encoding (What Real Implementations Do):
        ------------------------------------------------------
        1. Binary variables: x_{i,t} ∈ {0,1} for i ∈ cities, t ∈ time steps
           - x_{i,t} = 1 means "visit city i at position t in tour"
           - Total variables: O(n²)
        
        2. Constraints as penalties:
           a) Each city visited exactly once:
              P₁ · Σ_i (1 - Σ_t x_{i,t})²
           
           b) Each position has exactly one city:
              P₂ · Σ_t (1 - Σ_i x_{i,t})²
           
           where P₁, P₂ are large penalty coefficients
        
        3. Objective (minimize tour length):
              Σ_{i,j,t} d_{ij} · x_{i,t} · x_{j,t+1}
        
        4. Result: QUBO matrix size O(n²) × O(n²) = O(n⁴) elements!
        
        Simplified Approach Used Here:
        ------------------------------
        For demonstration, we create a smaller QUBO that captures the essence
        but is not the full encoding:
        
        - Use only n variables (one per city, representing visiting order)
        - Encode pairwise preferences based on distances
        - This is NOT sufficient for quantum solvers but illustrates the concept
        
        For real quantum TSP solving, use:
        - Specialized QAOA ansatz for TSP
        - D-Wave's problem embedding tools
        - Variational approaches with TSP-specific circuits
        
        Returns:
            Simplified QUBO matrix (n × n) - NOT full TSP encoding
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = TSPProblem(num_cities=5)
            >>> problem.generate()
            >>> 
            >>> # Get simplified QUBO (for demonstration only)
            >>> qubo = problem.to_qubo()
            >>> print(f"QUBO shape: {qubo.shape}")
            >>> 
            >>> # NOTE: This QUBO is simplified and should not be used
            >>> # for actual quantum TSP solving
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # SIMPLIFIED QUBO: Use distance matrix as basis
        # This is NOT the full TSP QUBO encoding!
        n = self.num_cities
        qubo = np.zeros((n, n), dtype=np.float64)
        
        # Fill with normalized distances (upper triangular)
        # Interpretation: Q[i][j] represents "preference" for i→j transition
        max_dist = np.max(self.distance_matrix)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Normalize distance to [0, 1] and use as QUBO coefficient
                normalized_dist = self.distance_matrix[i, j] / max_dist
                qubo[i, j] = normalized_dist
        
        # NOTE: This simplified QUBO does NOT encode:
        # - The constraint that each city is visited exactly once
        # - The constraint that the tour is connected
        # - The full O(n²) binary variable encoding
        #
        # Real TSP QUBO would be (n²) × (n²) matrix with penalty terms
        
        return qubo
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get TSP problem characteristics and metrics.
        
        Returns:
            Dictionary containing:
            - problem_type: 'tsp'
            - problem_size: Number of cities
            - complexity_class: 'NP-hard'
            - coordinate_range: Min/max coordinates used
            - avg_distance: Average pairwise distance
            - min_distance: Minimum pairwise distance
            - max_distance: Maximum pairwise distance
            - diameter: Maximum distance (graph diameter)
            - sparsity: 0.0 (TSP is complete graph, not sparse)
            - symmetry: True (distances are symmetric)
            - structure: 'complete' (fully connected)
            - estimated_classical_time: Estimated time for exact solution
            - estimated_quantum_advantage: Potential quantum speedup
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = TSPProblem(num_cities=30)
            >>> problem.generate()
            >>> metadata = problem.get_metadata()
            >>> 
            >>> print(f"Average distance: {metadata['avg_distance']:.2f}")
            >>> print(f"Coordinate spread: {metadata['coordinate_range']}")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Distance statistics
        # Extract upper triangle (exclude diagonal and duplicates)
        upper_triangle = self.distance_matrix[np.triu_indices(self.num_cities, k=1)]
        
        avg_distance = np.mean(upper_triangle)
        min_distance = np.min(upper_triangle)
        max_distance = np.max(upper_triangle)
        
        # Coordinate range
        coord_min = np.min(self.coordinates)
        coord_max = np.max(self.coordinates)
        
        # Estimated solution time
        # Exact: O(n!) - completely infeasible for n > 20
        # Held-Karp DP: O(n² · 2^n) - feasible up to ~25 cities
        # Practical heuristics (Lin-Kernighan): O(n² log n) or better
        if self.num_cities <= 15:
            classical_time_estimate = (np.math.factorial(self.num_cities) / 1e9)
        else:
            # Use Held-Karp estimate
            classical_time_estimate = (self.num_cities ** 2) * (2 ** self.num_cities) / 1e9
        
        # Quantum advantage estimate
        # TSP quantum advantage is still uncertain in practice
        # QAOA shows promise but needs more research
        if self.num_cities <= 10:
            quantum_advantage = 0.9  # Classical better for very small
        elif 10 < self.num_cities <= 30:
            quantum_advantage = 1.5  # Possible advantage for medium
        else:
            quantum_advantage = 1.2  # Uncertain for large
        
        return {
            'problem_type': self.problem_type,
            'problem_size': self.problem_size,
            'complexity_class': self.complexity_class,
            'coordinate_range': (float(coord_min), float(coord_max)),
            'avg_distance': float(avg_distance),
            'min_distance': float(min_distance),
            'max_distance': float(max_distance),
            'diameter': float(max_distance),
            'sparsity': 0.0,  # Complete graph - not sparse
            'symmetry': True,  # Euclidean distances are symmetric
            'structure': 'complete',
            'estimated_classical_time': classical_time_estimate,
            'estimated_quantum_advantage': quantum_advantage,
        }
    
    def get_optimal_solution_brute_force(self) -> Tuple[List[int], float]:
        """
        Find optimal TSP tour via exhaustive search.
        
        WARNING: Factorial time complexity O(n!). Only feasible for n <= 10.
        
        Algorithm:
        ----------
        1. Generate all permutations of cities (n! total)
        2. Evaluate tour distance for each permutation
        3. Return tour with minimum distance
        
        Optimization: Fix first city (reduces to (n-1)! permutations)
        
        Args:
            None
        
        Returns:
            Tuple of (optimal_tour, optimal_distance)
        
        Raises:
            ValueError: If problem not generated or n > 10 (too large)
        
        Example:
            >>> problem = TSPProblem(num_cities=8)
            >>> problem.generate(seed=42)
            >>> 
            >>> optimal_tour, optimal_dist = problem.get_optimal_solution_brute_force()
            >>> print(f"Optimal tour length: {optimal_dist:.2f}")
            >>> 
            >>> # Verify it's valid
            >>> assert problem.validate_solution(optimal_tour)
            >>> 
            >>> # Compare with random tour
            >>> random_tour = problem.get_random_solution()
            >>> random_dist = problem.calculate_cost(random_tour)
            >>> improvement = (random_dist - optimal_dist) / optimal_dist * 100
            >>> print(f"Random tour is {improvement:.1f}% longer")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        if self.num_cities > 10:
            raise ValueError(
                f"Brute force infeasible for n={self.num_cities} (too large). "
                f"Maximum supported: n=10 ({np.math.factorial(10):,} permutations)"
            )
        
        # Fix first city at 0 to reduce search space
        # This is valid because tour is cyclic (any rotation is equivalent)
        cities_to_permute = list(range(1, self.num_cities))
        
        best_tour = None
        best_distance = np.inf
        
        # Try all permutations
        for perm in permutations(cities_to_permute):
            # Construct tour starting from city 0
            tour = [0] + list(perm)
            
            # Calculate distance
            distance = self.calculate_cost(tour)
            
            # Update best
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
        
        return best_tour, best_distance
    
    def get_random_solution(self, seed: Optional[int] = None) -> List[int]:
        """
        Generate a random valid TSP tour.
        
        Creates a random permutation of cities, which is guaranteed to be
        a valid tour (visits each city exactly once).
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Random tour as list of city indices
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = TSPProblem(num_cities=20)
            >>> problem.generate()
            >>> 
            >>> # Get random tour
            >>> tour = problem.get_random_solution(seed=42)
            >>> assert problem.validate_solution(tour)
            >>> 
            >>> # Evaluate
            >>> distance = problem.calculate_cost(tour)
            >>> print(f"Random tour length: {distance:.2f}")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Random permutation of cities
        tour = list(np.random.permutation(self.num_cities))
        return tour
    
    def visualize(
        self,
        solution: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        show: bool = True,
    ) -> None:
        """
        Visualize TSP cities and tour.
        
        Creates a 2D plot showing:
        - City locations as numbered points
        - Tour edges connecting cities in order
        - Arrows showing tour direction
        - Total tour distance in title
        
        Args:
            solution: Tour to visualize (optional)
                     If None, shows only cities without tour
            save_path: Path to save figure (optional)
            figsize: Figure size in inches
            show: Whether to display the plot
        
        Raises:
            ValueError: If problem not generated or solution invalid
        
        Example:
            >>> problem = TSPProblem(num_cities=15)
            >>> problem.generate(seed=42)
            >>> 
            >>> # Visualize cities only
            >>> problem.visualize()
            >>> 
            >>> # Visualize with tour
            >>> tour = problem.get_random_solution()
            >>> problem.visualize(tour, save_path="tsp_tour.png")
            >>> 
            >>> # Find and visualize optimal (small problem)
            >>> if problem.num_cities <= 10:
            ...     opt_tour, _ = problem.get_optimal_solution_brute_force()
            ...     problem.visualize(opt_tour, save_path="optimal_tour.png")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        if solution is not None and not self.validate_solution(solution):
            raise ValueError("Invalid tour solution")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cities
        x_coords = self.coordinates[:, 0]
        y_coords = self.coordinates[:, 1]
        
        ax.scatter(x_coords, y_coords, c='red', s=200, zorder=3, alpha=0.8)
        
        # Add city labels
        for i in range(self.num_cities):
            ax.annotate(
                str(i),
                (x_coords[i], y_coords[i]),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                color='white',
                zorder=4
            )
        
        # Plot tour if provided
        if solution is not None:
            tour_distance = self.calculate_cost(solution)
            
            # Draw tour edges
            for i in range(self.num_cities):
                from_city = solution[i]
                to_city = solution[(i + 1) % self.num_cities]
                
                x1, y1 = self.coordinates[from_city]
                x2, y2 = self.coordinates[to_city]
                
                # Draw line
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6, zorder=1)
                
                # Draw arrow to show direction
                dx = x2 - x1
                dy = y2 - y1
                ax.arrow(
                    x1 + 0.3 * dx, y1 + 0.3 * dy,
                    0.2 * dx, 0.2 * dy,
                    head_width=2, head_length=2,
                    fc='blue', ec='blue', alpha=0.7, zorder=2
                )
            
            title = f"TSP Tour ({self.num_cities} cities)\nTotal Distance: {tour_distance:.2f}"
        else:
            title = f"TSP Cities ({self.num_cities} cities)"
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
