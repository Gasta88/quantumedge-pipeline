"""
MaxCut Problem Implementation.

MaxCut is a fundamental graph partitioning problem in combinatorial optimization:

Problem Definition:
    Given an undirected weighted graph G = (V, E), partition the vertex set V
    into two disjoint sets S and T such that the sum of weights of edges
    crossing the partition (cut edges) is maximized.
    
    Mathematically:
        MaxCut(G) = max_{S⊆V} Σ_{(u,v)∈E, u∈S, v∉S} w(u,v)
    
    where w(u,v) is the weight of edge (u,v).

Complexity:
    - NP-hard in general
    - Approximable within factor 0.878 (Goemans-Williamson algorithm)
    - Well-suited for quantum optimization (QAOA performs well)

Applications:
    - Circuit design (VLSI layout)
    - Statistical physics (Ising models)
    - Machine learning (clustering, semi-supervised learning)
    - Network analysis (community detection)

Quantum Advantage:
    MaxCut is one of the primary candidates for demonstrating quantum advantage:
    - Natural mapping to QAOA (Quantum Approximate Optimization Algorithm)
    - QUBO formulation directly maps to quantum Hamiltonians
    - Shows promise for near-term quantum devices (NISQ era)

Example Usage:
    >>> from src.problems.maxcut import MaxCutProblem
    >>> 
    >>> # Create MaxCut instance with 20 nodes
    >>> problem = MaxCutProblem(num_nodes=20)
    >>> 
    >>> # Generate random graph with 30% edge probability
    >>> problem.generate(seed=42, edge_probability=0.3, weight_range=(1.0, 10.0))
    >>> 
    >>> # Get a random partition
    >>> solution = problem.get_random_solution(seed=123)
    >>> 
    >>> # Validate and evaluate
    >>> assert problem.validate_solution(solution)
    >>> cost = problem.calculate_cost(solution)
    >>> print(f"Cut value: {-cost}")  # Negative because we minimize
    >>> 
    >>> # Get QUBO for quantum solver
    >>> qubo_matrix = problem.to_qubo()
    >>> 
    >>> # Visualize the partition
    >>> problem.visualize(solution, save_path="maxcut_solution.png")
    >>> 
    >>> # Get problem characteristics
    >>> metadata = problem.get_metadata()
    >>> print(f"Graph density: {metadata['graph_density']:.2f}")
    >>> print(f"Clustering: {metadata['clustering_coefficient']:.3f}")
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.problems.problem_base import ProblemBase


class MaxCutProblem(ProblemBase):
    """
    MaxCut graph partitioning problem.
    
    Represents a MaxCut optimization problem where the goal is to partition
    graph nodes into two sets to maximize the weight of edges between them.
    
    This implementation provides:
    - Random graph generation with configurable density
    - QUBO conversion for quantum solvers
    - Brute force solver for small instances (validation)
    - Visualization capabilities
    - Detailed problem characteristics for routing decisions
    
    Attributes:
        num_nodes (int): Number of nodes in the graph
        graph (nx.Graph): NetworkX graph representation (after generation)
        adjacency_matrix (np.ndarray): Weighted adjacency matrix
        edge_weights (Dict[Tuple[int, int], float]): Edge weight dictionary
    
    Example:
        >>> # Small problem with known optimal
        >>> problem = MaxCutProblem(num_nodes=5)
        >>> problem.generate(seed=42, edge_probability=0.6)
        >>> 
        >>> # Find optimal via brute force (feasible for n <= 15)
        >>> optimal_solution, optimal_value = problem.get_optimal_solution_brute_force()
        >>> print(f"Optimal cut value: {optimal_value}")
        >>> 
        >>> # Visualize optimal partition
        >>> problem.visualize(optimal_solution)
    """
    
    def __init__(self, num_nodes: int):
        """
        Initialize MaxCut problem instance.
        
        Args:
            num_nodes: Number of nodes in the graph (must be >= 2)
        
        Raises:
            ValueError: If num_nodes < 2
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=20)
            >>> print(problem)  # MAXCUT Problem (size=20, NP-hard)
        """
        super().__init__()
        
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        
        self._problem_type = "maxcut"
        self._problem_size = num_nodes
        self._complexity_class = "NP-hard"
        
        self.num_nodes = num_nodes
        self.graph: Optional[nx.Graph] = None
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.edge_weights: Dict[Tuple[int, int], float] = {}
    
    def generate(
        self,
        seed: Optional[int] = None,
        edge_probability: float = 0.5,
        weight_range: Tuple[float, float] = (1.0, 10.0),
        **kwargs
    ) -> None:
        """
        Generate a random weighted graph for MaxCut.
        
        Creates an Erdős-Rényi random graph G(n, p) where each edge appears
        independently with probability p. Edge weights are sampled uniformly
        from the specified range.
        
        Args:
            seed: Random seed for reproducibility
            edge_probability: Probability of edge existence (0 < p < 1)
                            - p=0.1: Sparse graph (~10% density)
                            - p=0.5: Medium density
                            - p=0.9: Dense graph (~90% density)
            weight_range: (min_weight, max_weight) for edge weights
            **kwargs: Additional arguments (ignored)
        
        Raises:
            ValueError: If parameters are out of valid ranges
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=10)
            >>> 
            >>> # Sparse graph with light weights
            >>> problem.generate(seed=42, edge_probability=0.2, weight_range=(1.0, 5.0))
            >>> 
            >>> # Dense graph with heavy weights
            >>> problem.generate(seed=123, edge_probability=0.8, weight_range=(5.0, 20.0))
        """
        # Validate parameters
        if not (0.0 < edge_probability < 1.0):
            raise ValueError("edge_probability must be in (0, 1)")
        
        if weight_range[0] <= 0 or weight_range[1] <= weight_range[0]:
            raise ValueError("weight_range must be (min, max) with 0 < min < max")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Generate Erdős-Rényi random graph
        self.graph = nx.erdos_renyi_graph(self.num_nodes, edge_probability, seed=seed)
        
        # Ensure graph is connected (for meaningful MaxCut)
        # If not connected, add random edges to connect components
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                # Connect consecutive components
                node1 = np.random.choice(list(components[i]))
                node2 = np.random.choice(list(components[i + 1]))
                self.graph.add_edge(node1, node2)
        
        # Assign random weights to edges
        min_weight, max_weight = weight_range
        for u, v in self.graph.edges():
            weight = np.random.uniform(min_weight, max_weight)
            self.graph[u][v]['weight'] = weight
            # Store in edge_weights dict for quick access
            self.edge_weights[(min(u, v), max(u, v))] = weight
        
        # Create adjacency matrix (weighted)
        self.adjacency_matrix = nx.to_numpy_array(self.graph, weight='weight')
        
        self._generated = True
    
    def validate_solution(self, solution: List[int]) -> bool:
        """
        Validate a MaxCut partition solution.
        
        A valid MaxCut solution is a binary assignment where each node is
        assigned to partition 0 or 1.
        
        Validation checks:
        1. Solution length matches number of nodes
        2. All values are binary (0 or 1)
        3. Problem has been generated
        
        Args:
            solution: Binary partition assignment [0, 1, 0, 1, ...]
                     solution[i] = 0 means node i is in partition S
                     solution[i] = 1 means node i is in partition T
        
        Returns:
            True if solution is valid, False otherwise
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=4)
            >>> problem.generate()
            >>> 
            >>> valid = [0, 1, 0, 1]
            >>> assert problem.validate_solution(valid)
            >>> 
            >>> invalid_length = [0, 1, 0]
            >>> assert not problem.validate_solution(invalid_length)
            >>> 
            >>> invalid_values = [0, 2, 0, 1]
            >>> assert not problem.validate_solution(invalid_values)
        """
        if not self._generated:
            return False
        
        # Check length
        if len(solution) != self.num_nodes:
            return False
        
        # Check all values are binary (0 or 1)
        if not all(x in [0, 1] for x in solution):
            return False
        
        return True
    
    def calculate_cost(self, solution: List[int]) -> float:
        """
        Calculate the MaxCut objective value (negative of cut weight).
        
        The cut value is the sum of weights of edges crossing the partition.
        An edge (u, v) is in the cut if u and v are in different partitions.
        
        Convention: Returns negative of cut value because we use minimization.
        To get actual cut value, negate the result: cut_value = -cost
        
        Mathematical formulation:
            cut_value = Σ_{(u,v)∈E} w(u,v) · (x_u ⊕ x_v)
        where:
            - w(u,v) is edge weight
            - x_u, x_v ∈ {0, 1} are partition assignments
            - ⊕ is XOR (1 if different partitions, 0 if same)
        
        Args:
            solution: Binary partition assignment
        
        Returns:
            Negative of cut value (for minimization)
        
        Raises:
            ValueError: If solution is invalid or problem not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=4)
            >>> problem.generate(seed=42)
            >>> solution = [0, 1, 0, 1]
            >>> cost = problem.calculate_cost(solution)
            >>> cut_value = -cost
            >>> print(f"Cut value: {cut_value:.2f}")
        """
        if not self.validate_solution(solution):
            raise ValueError("Invalid solution")
        
        cut_value = 0.0
        
        # Sum weights of edges crossing the partition
        for (u, v), weight in self.edge_weights.items():
            # Edge is cut if nodes are in different partitions (XOR)
            if solution[u] != solution[v]:
                cut_value += weight
        
        # Return negative for minimization convention
        return -cut_value
    
    def to_graph(self) -> nx.Graph:
        """
        Return the NetworkX graph representation.
        
        Returns the internal graph object with node and edge attributes.
        The graph has weighted edges stored in 'weight' attribute.
        
        Returns:
            NetworkX Graph with weighted edges
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=10)
            >>> problem.generate()
            >>> graph = problem.to_graph()
            >>> print(f"Nodes: {graph.number_of_nodes()}")
            >>> print(f"Edges: {graph.number_of_edges()}")
            >>> print(f"Density: {nx.density(graph):.3f}")
            >>> 
            >>> # Access edge weights
            >>> for u, v, data in graph.edges(data=True):
            ...     print(f"Edge ({u},{v}): weight={data['weight']:.2f}")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        return self.graph
    
    def to_qubo(self) -> np.ndarray:
        """
        Convert MaxCut to QUBO (Quadratic Unconstrained Binary Optimization) form.
        
        QUBO Formulation for MaxCut:
        ----------------------------
        We want to maximize: Σ_{(u,v)∈E} w(u,v) · (x_u ⊕ x_v)
        
        For binary variables x_u, x_v ∈ {0, 1}:
            x_u ⊕ x_v = x_u + x_v - 2·x_u·x_v
        
        Therefore, MaxCut objective:
            f(x) = Σ_{(u,v)∈E} w(u,v) · (x_u + x_v - 2·x_u·x_v)
        
        To convert maximization to minimization (QUBO standard):
            minimize -f(x) = Σ_{(u,v)∈E} w(u,v) · (2·x_u·x_v - x_u - x_v)
        
        QUBO Matrix Construction:
        -------------------------
        The QUBO matrix Q has:
        
        1. Diagonal terms Q[u][u] (linear coefficients):
            Q[u][u] = -Σ_{v: (u,v)∈E} w(u,v)
            (negative sum of weights of edges incident to u)
        
        2. Off-diagonal terms Q[u][v] for u < v (quadratic coefficients):
            Q[u][v] = 2·w(u,v) if edge (u,v) exists
            Q[u][v] = 0 otherwise
        
        Result: Upper-triangular matrix where minimizing x^T Q x gives MaxCut.
        
        Mathematical Verification:
            x^T Q x = Σ_u Q[u][u]·x_u + Σ_{u<v} Q[u][v]·x_u·x_v
                   = -Σ_u x_u·Σ_v w(u,v) + Σ_{u<v} 2·w(u,v)·x_u·x_v
                   = Σ_{(u,v)} w(u,v)·(2·x_u·x_v - x_u - x_v)
                   = -MaxCut objective
        
        Returns:
            Upper-triangular QUBO matrix (shape: n×n)
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=4)
            >>> problem.generate(seed=42)
            >>> qubo = problem.to_qubo()
            >>> 
            >>> # Verify it's upper triangular
            >>> assert np.allclose(qubo, np.triu(qubo))
            >>> 
            >>> # Diagonal contains negative degree (weighted)
            >>> print("Linear terms (diagonal):", np.diag(qubo))
            >>> 
            >>> # Evaluate a solution using QUBO
            >>> solution = np.array([0, 1, 0, 1])
            >>> qubo_value = solution @ qubo @ solution
            >>> direct_value = problem.calculate_cost(solution.tolist())
            >>> assert np.isclose(qubo_value, direct_value)
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        n = self.num_nodes
        qubo = np.zeros((n, n), dtype=np.float64)
        
        # Construct QUBO matrix
        for (u, v), weight in self.edge_weights.items():
            # Off-diagonal: Q[u][v] = 2 * w(u,v) for u < v
            qubo[u, v] = 2.0 * weight
            
            # Diagonal: Q[u][u] -= w(u,v) and Q[v][v] -= w(u,v)
            qubo[u, u] -= weight
            qubo[v, v] -= weight
        
        # Ensure upper triangular (should already be, but for safety)
        qubo = np.triu(qubo)
        
        return qubo
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get MaxCut problem characteristics and metrics.
        
        Provides detailed information about the problem instance for
        routing decisions and performance analysis.
        
        Returns:
            Dictionary containing:
            - problem_type: 'maxcut'
            - problem_size: Number of nodes
            - complexity_class: 'NP-hard'
            - num_edges: Number of edges
            - graph_density: Edge density (0-1)
            - sparsity: 1 - density (for consistency with other problems)
            - clustering_coefficient: Average clustering
            - average_degree: Mean node degree
            - max_degree: Maximum node degree
            - diameter: Graph diameter (max shortest path)
            - total_weight: Sum of all edge weights
            - average_weight: Mean edge weight
            - structure: Graph structure classification
            - symmetry: True (MaxCut graph is always symmetric)
            - estimated_classical_time: Estimated time for classical solver (s)
            - estimated_quantum_advantage: Potential quantum speedup ratio
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=50)
            >>> problem.generate(edge_probability=0.2)
            >>> metadata = problem.get_metadata()
            >>> 
            >>> # Use for routing decision
            >>> if metadata['sparsity'] > 0.7 and metadata['problem_size'] > 30:
            ...     print("Route to quantum: sparse, medium-large problem")
            >>> 
            >>> print(f"Graph has {metadata['num_edges']} edges")
            >>> print(f"Density: {metadata['graph_density']:.3f}")
            >>> print(f"Clustering: {metadata['clustering_coefficient']:.3f}")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Basic graph metrics
        num_edges = self.graph.number_of_edges()
        max_possible_edges = self.num_nodes * (self.num_nodes - 1) / 2
        graph_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Degree statistics
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        
        # Clustering coefficient
        clustering = nx.average_clustering(self.graph)
        
        # Diameter (maximum shortest path)
        try:
            diameter = nx.diameter(self.graph)
        except nx.NetworkXError:
            # Graph not connected
            diameter = -1
        
        # Weight statistics
        total_weight = sum(self.edge_weights.values())
        avg_weight = total_weight / num_edges if num_edges > 0 else 0
        
        # Classify graph structure
        if graph_density < 0.2:
            structure = "sparse"
        elif graph_density < 0.5:
            structure = "medium"
        elif graph_density < 0.8:
            structure = "dense"
        else:
            structure = "very_dense"
        
        # Estimate classical solver time (heuristic)
        # Goemans-Williamson SDP solver: O(n^3.5) roughly
        classical_time_estimate = (self.num_nodes ** 3.5) / 1e9  # Rough estimate
        
        # Estimate quantum advantage
        # QAOA shows advantage for medium-sized sparse graphs
        # Advantage is higher for sparse graphs with medium size
        if 20 <= self.num_nodes <= 100 and graph_density < 0.3:
            quantum_advantage = 2.0 + (1.0 - graph_density)  # Higher for sparser
        elif self.num_nodes < 20:
            quantum_advantage = 0.8  # Classical better for small
        else:
            quantum_advantage = 1.2  # Modest advantage for large/dense
        
        return {
            'problem_type': self.problem_type,
            'problem_size': self.problem_size,
            'complexity_class': self.complexity_class,
            'num_edges': num_edges,
            'graph_density': graph_density,
            'sparsity': 1.0 - graph_density,
            'clustering_coefficient': clustering,
            'average_degree': avg_degree,
            'max_degree': max_degree,
            'diameter': diameter,
            'total_weight': total_weight,
            'average_weight': avg_weight,
            'structure': structure,
            'symmetry': True,  # MaxCut graph is always undirected/symmetric
            'estimated_classical_time': classical_time_estimate,
            'estimated_quantum_advantage': quantum_advantage,
        }
    
    def get_optimal_solution_brute_force(self) -> Tuple[List[int], float]:
        """
        Find optimal MaxCut solution via exhaustive search.
        
        WARNING: Exponential time complexity O(2^n). Only feasible for n <= 15.
        This method is useful for:
        - Validating heuristic solvers on small instances
        - Benchmarking quantum algorithms
        - Testing correctness of QUBO conversion
        
        Algorithm:
        ----------
        1. Enumerate all 2^n possible partitions
        2. Evaluate cut value for each partition
        3. Return partition with maximum cut value
        
        Args:
            None
        
        Returns:
            Tuple of (optimal_solution, optimal_cut_value)
            where optimal_solution is binary partition and
            optimal_cut_value is the maximum cut weight
        
        Raises:
            ValueError: If problem not generated or n > 15 (too large)
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=8)
            >>> problem.generate(seed=42)
            >>> 
            >>> optimal_sol, optimal_val = problem.get_optimal_solution_brute_force()
            >>> print(f"Optimal cut: {optimal_val:.2f}")
            >>> 
            >>> # Verify it's optimal
            >>> cost = problem.calculate_cost(optimal_sol)
            >>> assert abs(-cost - optimal_val) < 1e-6
            >>> 
            >>> # Compare with heuristic
            >>> random_sol = problem.get_random_solution()
            >>> random_val = -problem.calculate_cost(random_sol)
            >>> print(f"Random solution: {random_val:.2f}")
            >>> print(f"Optimality gap: {(optimal_val - random_val) / optimal_val * 100:.1f}%")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        if self.num_nodes > 15:
            raise ValueError(
                f"Brute force infeasible for n={self.num_nodes} (too large). "
                "Maximum supported: n=15 (32768 partitions)"
            )
        
        best_solution = None
        best_cut_value = -np.inf
        
        # Enumerate all 2^n partitions
        for partition_bits in range(2 ** self.num_nodes):
            # Convert integer to binary partition
            solution = [
                (partition_bits >> i) & 1
                for i in range(self.num_nodes)
            ]
            
            # Evaluate cut value
            cost = self.calculate_cost(solution)
            cut_value = -cost  # Negate because calculate_cost minimizes
            
            # Update best
            if cut_value > best_cut_value:
                best_cut_value = cut_value
                best_solution = solution
        
        return best_solution, best_cut_value
    
    def get_problem_features(self) -> Dict[str, float]:
        """
        Extract graph-theoretic features for machine learning or analysis.
        
        Computes additional graph features beyond get_metadata() that may
        be useful for predictive models or detailed analysis.
        
        Returns:
            Dictionary of graph features:
            - degree_centrality_mean: Average degree centrality
            - degree_centrality_std: Std dev of degree centrality
            - betweenness_centrality_mean: Average betweenness
            - closeness_centrality_mean: Average closeness
            - edge_weight_std: Std dev of edge weights
            - assortativity: Degree assortativity coefficient
            - transitivity: Global clustering coefficient
        
        Raises:
            ValueError: If problem not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate(seed=42)
            >>> features = problem.get_problem_features()
            >>> 
            >>> # Use for analysis
            >>> print(f"Degree heterogeneity: {features['degree_centrality_std']:.3f}")
            >>> print(f"Assortativity: {features['assortativity']:.3f}")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        # Centrality measures
        degree_cent = list(nx.degree_centrality(self.graph).values())
        betweenness_cent = list(nx.betweenness_centrality(self.graph).values())
        closeness_cent = list(nx.closeness_centrality(self.graph).values())
        
        # Edge weight statistics
        weights = list(self.edge_weights.values())
        
        # Assortativity (correlation of node degrees)
        try:
            assortativity = nx.degree_assortativity_coefficient(self.graph)
        except:
            assortativity = 0.0
        
        # Transitivity (global clustering)
        transitivity = nx.transitivity(self.graph)
        
        return {
            'degree_centrality_mean': np.mean(degree_cent),
            'degree_centrality_std': np.std(degree_cent),
            'betweenness_centrality_mean': np.mean(betweenness_cent),
            'closeness_centrality_mean': np.mean(closeness_cent),
            'edge_weight_std': np.std(weights),
            'assortativity': assortativity,
            'transitivity': transitivity,
        }
    
    def visualize(
        self,
        solution: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        node_size: int = 500,
        show: bool = True,
    ) -> None:
        """
        Visualize the MaxCut graph with optional solution partition coloring.
        
        Creates a visual representation of the graph with:
        - Nodes colored by partition (if solution provided)
        - Edges colored by whether they're cut (red) or not (gray)
        - Edge thickness proportional to weight
        - Node labels
        
        Args:
            solution: Binary partition to visualize (optional)
                     If None, shows graph without partitioning
            save_path: Path to save figure (optional)
                      If None, only displays
            figsize: Figure size in inches (width, height)
            node_size: Size of nodes in plot
            show: Whether to display the plot (plt.show())
        
        Raises:
            ValueError: If problem not generated or solution invalid
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=10)
            >>> problem.generate(seed=42)
            >>> 
            >>> # Visualize without solution
            >>> problem.visualize()
            >>> 
            >>> # Visualize with partition
            >>> solution = problem.get_random_solution()
            >>> problem.visualize(solution, save_path="maxcut_partition.png")
            >>> 
            >>> # Find and visualize optimal (small graph)
            >>> if problem.num_nodes <= 10:
            ...     opt_sol, _ = problem.get_optimal_solution_brute_force()
            ...     problem.visualize(opt_sol, save_path="optimal_partition.png")
        """
        if not self._generated:
            raise ValueError("Problem not generated. Call generate() first.")
        
        if solution is not None and not self.validate_solution(solution):
            raise ValueError("Invalid solution provided")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute layout (spring layout works well for most graphs)
        pos = nx.spring_layout(self.graph, seed=42, k=1/np.sqrt(self.num_nodes))
        
        if solution is None:
            # Visualize graph without partitioning
            node_colors = ['lightblue'] * self.num_nodes
            edge_colors = ['gray'] * self.graph.number_of_edges()
            title = f"MaxCut Graph ({self.num_nodes} nodes, {self.graph.number_of_edges()} edges)"
        else:
            # Color nodes by partition
            node_colors = ['lightcoral' if solution[i] == 0 else 'lightgreen'
                          for i in range(self.num_nodes)]
            
            # Color edges: red if cut, gray if not cut
            edge_colors = []
            for u, v in self.graph.edges():
                if solution[u] != solution[v]:
                    edge_colors.append('red')
                else:
                    edge_colors.append('lightgray')
            
            # Calculate cut value
            cut_value = -self.calculate_cost(solution)
            title = f"MaxCut Partition (Cut Value: {cut_value:.2f})"
        
        # Get edge weights for thickness
        edge_weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [2.0 * (w / max_weight) for w in edge_weights]
        
        # Draw graph
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_size,
            alpha=0.9,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # Add legend if showing partition
        if solution is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightcoral', label='Partition 0'),
                Patch(facecolor='lightgreen', label='Partition 1'),
                Patch(facecolor='red', label='Cut edges'),
                Patch(facecolor='lightgray', label='Non-cut edges'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
