"""
Problem Analyzer for QuantumEdge Pipeline.

This module provides comprehensive analysis of optimization problems to support
intelligent routing decisions between classical and quantum solvers. The analyzer
extracts problem features, estimates computational requirements, and predicts
quantum advantage potential.

Key Capabilities
----------------
1. **Feature Extraction**: Analyzes problem structure and characteristics
2. **Runtime Estimation**: Predicts classical and quantum solver performance
3. **Quantum Advantage Prediction**: Estimates likelihood of quantum speedup
4. **Graph Analysis**: Computes graph-theoretic properties for graph-based problems

The analyzer helps the router make informed decisions by quantifying:
- Problem complexity and size
- Structural properties (density, clustering, connectivity)
- Computational requirements for different solver types
- Expected performance gains from quantum approaches

Example Usage
-------------
```python
from src.problems.maxcut import MaxCutProblem
from src.analyzer.problem_analyzer import ProblemAnalyzer

# Create and generate problem
problem = MaxCutProblem(num_nodes=50)
problem.generate(edge_probability=0.2)

# Analyze problem characteristics
analyzer = ProblemAnalyzer()
analysis = analyzer.analyze_problem(problem)

# Check results
print(f"Problem size: {analysis['problem_size']}")
print(f"Graph density: {analysis['graph_features']['density']:.3f}")
print(f"Classical score: {analysis['suitability_scores']['classical_score']:.2f}")
print(f"Quantum score: {analysis['suitability_scores']['quantum_score']:.2f}")
print(f"Quantum advantage probability: {analyzer.predict_quantum_advantage(problem):.2%}")

# Use for routing decision
if analysis['suitability_scores']['quantum_score'] > 0.6:
    print("Route to quantum solver")
else:
    print("Route to classical solver")
```

Design Philosophy
-----------------
The analyzer uses heuristic-based rules for now, designed to be easily
upgraded to ML-based predictions later. All analysis methods are modular
and can be enhanced independently without breaking the interface.
"""

from typing import Dict, Any, Optional
import logging
import numpy as np
import networkx as nx

from src.problems.problem_base import ProblemBase


# Configure module logger
logger = logging.getLogger(__name__)


class ProblemAnalyzer:
    """
    Analyzes optimization problems to extract features and predict solver performance.
    
    This class provides comprehensive problem analysis to support intelligent routing
    decisions. It extracts structural features, estimates computational requirements,
    and predicts quantum advantage potential using heuristic rules.
    
    The analyzer is designed to work with any problem implementing the ProblemBase
    interface, making it problem-agnostic and easily extensible.
    
    Attributes:
        None (stateless design - all methods take problem as input)
    
    Thread Safety:
        This class is stateless and thread-safe. Multiple threads can share
        a single ProblemAnalyzer instance.
    
    Performance Notes:
        - Graph analysis (clustering, diameter) can be expensive for large graphs
        - Results are computed fresh on each call (no caching)
        - For repeated analysis, consider caching results externally
    
    Future Enhancements:
        - Add ML-based runtime prediction models
        - Incorporate historical solver performance data
        - Support custom feature extractors
        - Add problem similarity metrics
    """
    
    def __init__(self):
        """
        Initialize the problem analyzer.
        
        The analyzer is stateless by design - all analysis is performed
        based on the problem instance passed to analysis methods.
        """
        logger.info("ProblemAnalyzer initialized")
    
    def analyze_problem(self, problem: ProblemBase) -> Dict[str, Any]:
        """
        Extract comprehensive features from a problem instance.
        
        This is the main entry point for problem analysis. It extracts:
        - Basic problem properties (type, size, complexity)
        - Graph-theoretic features (for graph-based problems)
        - Suitability scores for classical vs quantum solvers
        - Runtime estimates
        
        The analysis results can be used by the router to make informed
        solver selection decisions.
        
        Args:
            problem: A problem instance implementing ProblemBase interface.
                    Must be generated (problem.is_generated == True).
        
        Returns:
            Dictionary containing:
            {
                'problem_type': str,           # 'maxcut', 'tsp', 'portfolio'
                'problem_size': int,           # Number of variables
                'complexity_estimate': str,    # 'small', 'medium', 'large', 'very_large'
                'graph_features': {            # Only for graph-based problems
                    'density': float,          # 0-1, edge density
                    'clustering_coefficient': float,  # 0-1, transitivity
                    'average_degree': float,   # Average node degree
                    'diameter': int or None    # Graph diameter (None if too expensive)
                },
                'suitability_scores': {
                    'classical_score': float,  # 0-1, classical solver suitability
                    'quantum_score': float     # 0-1, quantum solver suitability
                },
                'estimated_classical_runtime': float,  # seconds
                'estimated_quantum_runtime': float,    # seconds
                'quantum_advantage_probability': float  # 0-1, P(quantum faster)
            }
        
        Raises:
            ValueError: If problem is not generated
            TypeError: If problem doesn't implement ProblemBase
        
        Example:
            >>> from src.problems.maxcut import MaxCutProblem
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate(edge_probability=0.3)
            >>> 
            >>> analyzer = ProblemAnalyzer()
            >>> analysis = analyzer.analyze_problem(problem)
            >>> 
            >>> print(f"Type: {analysis['problem_type']}")
            >>> print(f"Size: {analysis['problem_size']}")
            >>> print(f"Density: {analysis['graph_features']['density']:.2f}")
            >>> print(f"Classical score: {analysis['suitability_scores']['classical_score']:.2f}")
        
        Notes:
            - Graph diameter computation is skipped for graphs with >100 nodes (too expensive)
            - All scores are normalized to [0, 1] range
            - Runtime estimates are rough approximations, not precise predictions
        """
        # Validate input
        if not isinstance(problem, ProblemBase):
            raise TypeError(f"Problem must implement ProblemBase, got {type(problem)}")
        
        if not problem.is_generated:
            raise ValueError("Problem must be generated before analysis. Call problem.generate() first.")
        
        logger.info(f"Analyzing problem: {problem}")
        
        # Extract basic properties
        problem_type = problem.problem_type
        problem_size = problem.problem_size
        complexity_class = problem.complexity_class
        
        # Categorize problem size for routing decisions
        complexity_estimate = self._estimate_complexity_category(problem_size)
        
        logger.debug(f"Basic properties - type: {problem_type}, size: {problem_size}, "
                    f"complexity: {complexity_estimate}")
        
        # Extract graph features (if applicable)
        graph_features = {}
        try:
            graph = problem.to_graph()
            graph_features = self._extract_graph_features(graph)
            logger.debug(f"Graph features extracted: density={graph_features.get('density', 0):.3f}")
        except Exception as e:
            logger.warning(f"Could not extract graph features: {e}")
            graph_features = None
        
        # Calculate suitability scores
        suitability_scores = self._calculate_suitability_scores(
            problem_type=problem_type,
            problem_size=problem_size,
            complexity_class=complexity_class,
            graph_features=graph_features
        )
        
        logger.debug(f"Suitability scores - classical: {suitability_scores['classical_score']:.2f}, "
                    f"quantum: {suitability_scores['quantum_score']:.2f}")
        
        # Estimate runtimes
        classical_runtime = self.estimate_classical_runtime(problem)
        quantum_runtime = self.estimate_quantum_runtime(problem)
        
        logger.debug(f"Runtime estimates - classical: {classical_runtime:.3f}s, "
                    f"quantum: {quantum_runtime:.3f}s")
        
        # Predict quantum advantage
        quantum_advantage_prob = self.predict_quantum_advantage(problem)
        
        logger.info(f"Analysis complete - quantum advantage probability: {quantum_advantage_prob:.2%}")
        
        # Compile results
        analysis = {
            'problem_type': problem_type,
            'problem_size': problem_size,
            'complexity_estimate': complexity_estimate,
            'graph_features': graph_features,
            'suitability_scores': suitability_scores,
            'estimated_classical_runtime': classical_runtime,
            'estimated_quantum_runtime': quantum_runtime,
            'quantum_advantage_probability': quantum_advantage_prob
        }
        
        return analysis
    
    def estimate_classical_runtime(self, problem: ProblemBase) -> float:
        """
        Estimate classical solver runtime based on problem characteristics.
        
        Uses heuristics based on:
        - Problem type (different algorithms have different complexities)
        - Problem size (scaling behavior)
        - Historical benchmarks (rough averages)
        
        These estimates are intentionally conservative (upper bounds) to
        avoid underestimating runtime and making poor routing decisions.
        
        Complexity Assumptions:
        -----------------------
        MaxCut:
            - Small (n≤20): Exact solver possible, O(2^n) ~ 0.01-1s
            - Medium (20<n≤50): Approximation algorithms, O(n²) ~ 0.1-10s
            - Large (n>50): Heuristics required, O(n² log n) ~ 1-100s
        
        TSP:
            - Small (n≤15): Exact DP, O(n² 2^n) ~ 0.1-10s
            - Medium (15<n≤50): Branch & bound, O(n!) ~ 10-300s
            - Large (n>50): Heuristics (Christofides), O(n³) ~ 1-100s
        
        Portfolio Optimization:
            - Generally efficient: Convex optimization O(n³) ~ 0.001-1s
            - With constraints: May need MILP, can be slower
        
        Args:
            problem: Problem instance to estimate runtime for
        
        Returns:
            Estimated runtime in seconds (float)
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=40)
            >>> problem.generate()
            >>> analyzer = ProblemAnalyzer()
            >>> runtime = analyzer.estimate_classical_runtime(problem)
            >>> print(f"Estimated classical runtime: {runtime:.2f}s")
        
        Notes:
            - These are rough estimates, actual runtime can vary significantly
            - Assumes modern hardware (multi-core CPU, sufficient RAM)
            - Does not account for problem structure (all instances treated similarly)
            - Conservative estimates to avoid underestimation
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before runtime estimation")
        
        problem_type = problem.problem_type.lower()
        size = problem.problem_size
        
        logger.debug(f"Estimating classical runtime for {problem_type} problem of size {size}")
        
        # Type-specific runtime estimation
        if problem_type == 'maxcut':
            if size <= 20:
                # Small: can solve exactly with branch & bound or SDP
                runtime = 0.01 * (2 ** (size / 10))  # Exponential scaling
            elif size <= 50:
                # Medium: use approximation algorithms (SDP relaxation)
                runtime = 0.001 * (size ** 2)  # Quadratic scaling
            else:
                # Large: use fast heuristics (greedy, local search)
                runtime = 0.001 * size * np.log(size)  # O(n log n)
        
        elif problem_type == 'tsp':
            if size <= 15:
                # Small: exact dynamic programming (Held-Karp)
                runtime = 0.001 * (size ** 2) * (2 ** size)  # O(n² 2^n)
            elif size <= 50:
                # Medium: branch & bound with pruning
                runtime = 1.0 * (size ** 3)  # Pessimistic estimate
            else:
                # Large: heuristics (Christofides, Lin-Kernighan)
                runtime = 0.01 * (size ** 2)  # O(n²) for good heuristics
        
        elif problem_type == 'portfolio':
            # Generally efficient: quadratic programming
            # Scales well even for large portfolios
            runtime = 0.0001 * (size ** 3)  # O(n³) for interior point methods
            
            # Add penalty for highly constrained problems
            if size > 100:
                runtime *= 1.5  # Additional overhead for large-scale problems
        
        else:
            # Unknown problem type: use generic scaling
            logger.warning(f"Unknown problem type '{problem_type}', using generic estimate")
            runtime = 0.01 * (size ** 2)
        
        # Apply minimum runtime (overhead for problem loading, validation)
        runtime = max(runtime, 0.001)
        
        logger.debug(f"Classical runtime estimate: {runtime:.4f}s")
        
        return runtime
    
    def estimate_quantum_runtime(self, problem: ProblemBase) -> float:
        """
        Estimate quantum simulator runtime.
        
        Quantum runtime depends on:
        - Number of qubits required (determines state space size: 2^n)
        - Circuit depth (number of gate layers)
        - Number of shots (measurements for statistics)
        - Simulator overhead (exponential in qubits)
        
        NOTE: This estimates SIMULATOR runtime, not real quantum hardware.
        Real quantum computers would be much faster but have limited qubit counts
        and high error rates currently.
        
        Simulator Complexity:
        ---------------------
        - State vector simulation: O(2^n) memory, O(2^n × depth) time
        - Practical limit: ~25-30 qubits on standard workstation
        - Each additional qubit doubles memory and runtime
        
        QAOA Circuit Structure (typical):
        ----------------------------------
        - Qubits needed: n (one per problem variable)
        - Circuit depth: p × (problem_layers + mixer_layers)
          * p = QAOA parameter layers (typically 1-5)
          * problem_layers ≈ number of QUBO terms / n
          * mixer_layers = n (X rotations)
        - Shots: 1000-10000 for statistical accuracy
        
        Args:
            problem: Problem instance to estimate runtime for
        
        Returns:
            Estimated simulator runtime in seconds (float)
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=20)
            >>> problem.generate()
            >>> analyzer = ProblemAnalyzer()
            >>> runtime = analyzer.estimate_quantum_runtime(problem)
            >>> print(f"Estimated quantum simulator runtime: {runtime:.2f}s")
        
        Notes:
            - These estimates are for SIMULATION, not real quantum hardware
            - Real quantum computers would be faster but are limited by:
              * Qubit count (current systems: 50-1000 qubits)
              * Error rates (requiring error correction overhead)
              * Queue times and calibration
            - Simulator runtime grows exponentially with qubit count
            - For >30 qubits, simulation becomes impractical on classical hardware
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before runtime estimation")
        
        size = problem.problem_size
        
        logger.debug(f"Estimating quantum runtime for problem of size {size}")
        
        # Estimate circuit parameters
        num_qubits = size  # Typically 1 qubit per variable
        
        # Estimate circuit depth based on problem structure
        # QAOA: depth = p × (C + M), where C=problem, M=mixer
        # Assume p=3 layers, C≈n, M=n → depth ≈ 6n
        qaoa_layers = 3
        circuit_depth = qaoa_layers * size * 2
        
        # Number of measurements for statistical accuracy
        num_shots = 1000
        
        logger.debug(f"Circuit parameters - qubits: {num_qubits}, depth: {circuit_depth}, "
                    f"shots: {num_shots}")
        
        # Simulator time estimation
        # Base time per shot: exponential in qubits
        time_per_shot = 0.0001 * (2 ** (num_qubits / 10))  # Grows exponentially
        
        # Total execution time
        execution_time = time_per_shot * circuit_depth * num_shots
        
        # Add compilation overhead (circuit optimization, transpilation)
        compilation_time = 1.0 + (0.1 * num_qubits)
        
        # Total runtime
        runtime = execution_time + compilation_time
        
        # For very large problems, simulation becomes impractical
        if num_qubits > 25:
            # Add penalty to reflect difficulty of simulation
            penalty = 2 ** (num_qubits - 25)
            runtime *= penalty
            logger.warning(f"Large qubit count ({num_qubits}) - simulation may be impractical")
        
        logger.debug(f"Quantum runtime estimate: {runtime:.4f}s "
                    f"(execution: {execution_time:.4f}s, compilation: {compilation_time:.4f}s)")
        
        return runtime
    
    def predict_quantum_advantage(self, problem: ProblemBase) -> float:
        """
        Predict probability that quantum solver will be faster than classical.
        
        Uses heuristic rules based on:
        - Problem type (some problems benefit more from quantum)
        - Problem size (quantum advantage appears in certain size ranges)
        - Problem structure (graph properties affect quantum performance)
        - Historical benchmarks (literature results)
        
        Heuristic Rules:
        ----------------
        1. **Problem Type**:
           - MaxCut, Graph Coloring: High potential (proven QAOA advantage)
           - TSP: Medium potential (mixed results)
           - Portfolio: Low potential (classical convex opt is very efficient)
        
        2. **Problem Size**:
           - Too small (n<10): Classical faster (quantum overhead dominates)
           - Sweet spot (10≤n≤50): Potential quantum advantage
           - Too large (n>50): Simulation impractical, but real QC could help
        
        3. **Graph Structure** (for graph problems):
           - Sparse graphs (density<0.3): Better for quantum (fewer interactions)
           - High clustering: Classical algorithms exploit locality better
           - Regular structure: Quantum can find patterns efficiently
        
        4. **Complexity Class**:
           - NP-hard: Higher quantum potential
           - P: Classical algorithms are optimal
        
        Score Calculation:
        ------------------
        Start with base score, then apply multipliers for each factor.
        Final score is probability quantum will be faster (0-1 scale).
        
        Args:
            problem: Problem instance to predict advantage for
        
        Returns:
            Probability quantum will be faster (0.0 - 1.0)
            - 0.0-0.3: Classical strongly preferred
            - 0.3-0.5: Classical slightly preferred
            - 0.5-0.7: Quantum slightly preferred
            - 0.7-1.0: Quantum strongly preferred
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate(edge_probability=0.2)
            >>> analyzer = ProblemAnalyzer()
            >>> prob = analyzer.predict_quantum_advantage(problem)
            >>> print(f"Quantum advantage probability: {prob:.2%}")
            >>> 
            >>> if prob > 0.5:
            ...     print("Recommend quantum solver")
            ... else:
            ...     print("Recommend classical solver")
        
        Notes:
            - This is a heuristic prediction, not a guarantee
            - Based on theoretical analysis and empirical benchmarks
            - Real quantum hardware performance may differ from simulation
            - Consider this as one factor in routing decisions, not the only factor
            - Can be upgraded to ML-based prediction with historical data
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before prediction")
        
        problem_type = problem.problem_type.lower()
        size = problem.problem_size
        complexity_class = problem.complexity_class.lower()
        
        logger.debug(f"Predicting quantum advantage for {problem_type} problem (size={size})")
        
        # Start with base score
        score = 0.5  # Neutral starting point
        
        # Factor 1: Problem type
        # Some problems are known to benefit more from quantum approaches
        problem_type_scores = {
            'maxcut': 0.7,      # Strong quantum advantage demonstrated
            'tsp': 0.5,         # Mixed results, problem-dependent
            'portfolio': 0.3,   # Classical convex optimization is very efficient
            'graph_coloring': 0.6,  # Quantum annealing shows promise
            'sat': 0.6,         # QAOA can find satisfying assignments efficiently
        }
        
        type_bonus = problem_type_scores.get(problem_type, 0.5)
        score = score * 0.4 + type_bonus * 0.6  # Weight toward problem type
        
        logger.debug(f"After problem type factor: {score:.3f}")
        
        # Factor 2: Problem size
        # Quantum advantage appears in specific size ranges
        if size < 10:
            # Too small: quantum overhead dominates
            size_multiplier = 0.5
        elif 10 <= size <= 30:
            # Sweet spot: quantum can show advantage
            size_multiplier = 1.3
        elif 30 < size <= 50:
            # Still good, but classical algorithms scale well too
            size_multiplier = 1.1
        else:
            # Large: simulation impractical, but real QC could help
            size_multiplier = 0.8
        
        score *= size_multiplier
        logger.debug(f"After size factor: {score:.3f}")
        
        # Factor 3: Graph structure (if applicable)
        try:
            graph = problem.to_graph()
            density = nx.density(graph)
            
            # Sparse graphs favor quantum (fewer entangling gates needed)
            if density < 0.2:
                structure_multiplier = 1.2
            elif density < 0.5:
                structure_multiplier = 1.0
            else:
                # Dense graphs: classical algorithms can be more efficient
                structure_multiplier = 0.8
            
            score *= structure_multiplier
            logger.debug(f"After structure factor (density={density:.3f}): {score:.3f}")
            
            # High clustering: classical algorithms exploit locality
            try:
                clustering = nx.average_clustering(graph)
                if clustering > 0.5:
                    score *= 0.9  # Slight penalty
                logger.debug(f"After clustering factor (clustering={clustering:.3f}): {score:.3f}")
            except:
                pass  # Skip if clustering computation fails
        
        except Exception as e:
            logger.debug(f"Could not analyze graph structure: {e}")
        
        # Factor 4: Complexity class
        # NP-hard problems have higher quantum potential
        if 'np' in complexity_class and 'hard' in complexity_class:
            score *= 1.1
            logger.debug(f"After complexity factor (NP-hard): {score:.3f}")
        elif complexity_class == 'p':
            score *= 0.8
            logger.debug(f"After complexity factor (P): {score:.3f}")
        
        # Clamp to [0, 1] range
        score = max(0.0, min(1.0, score))
        
        logger.info(f"Quantum advantage probability: {score:.3f}")
        
        return score
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _estimate_complexity_category(self, size: int) -> str:
        """
        Categorize problem size into complexity categories.
        
        These categories are used for routing decisions and resource allocation:
        - 'small': Can be solved exactly, fast execution
        - 'medium': Requires approximation, moderate resources
        - 'large': Heuristics needed, significant resources
        - 'very_large': May need distributed computing or specialized hardware
        
        Args:
            size: Problem size (number of variables)
        
        Returns:
            Complexity category string
        """
        if size <= 20:
            return 'small'
        elif size <= 50:
            return 'medium'
        elif size <= 200:
            return 'large'
        else:
            return 'very_large'
    
    def _extract_graph_features(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Extract graph-theoretic features from a NetworkX graph.
        
        Computes:
        - density: Ratio of actual edges to possible edges
        - clustering_coefficient: Measure of local connectivity (transitivity)
        - average_degree: Average number of connections per node
        - diameter: Longest shortest path (skipped for large graphs)
        
        These features help characterize problem structure and predict
        solver performance.
        
        Args:
            graph: NetworkX graph to analyze
        
        Returns:
            Dictionary of graph features
        """
        features = {}
        
        # Density: How many edges exist vs how many possible
        # density = 0: no edges (disconnected)
        # density = 1: complete graph (all possible edges)
        density = nx.density(graph)
        features['density'] = density
        
        # Clustering coefficient: Measure of transitivity
        # High clustering → nodes form tight communities
        # Low clustering → sparse, tree-like structure
        try:
            clustering = nx.average_clustering(graph)
            features['clustering_coefficient'] = clustering
        except:
            features['clustering_coefficient'] = 0.0
        
        # Average degree: Average number of neighbors per node
        # Useful for estimating problem difficulty
        if graph.number_of_nodes() > 0:
            degrees = [d for n, d in graph.degree()]
            average_degree = sum(degrees) / len(degrees)
            features['average_degree'] = average_degree
        else:
            features['average_degree'] = 0.0
        
        # Diameter: Longest shortest path in graph
        # Expensive to compute for large graphs, so we skip it
        # High diameter → elongated structure
        # Low diameter → compact, well-connected
        if graph.number_of_nodes() <= 100 and nx.is_connected(graph):
            try:
                diameter = nx.diameter(graph)
                features['diameter'] = diameter
            except:
                features['diameter'] = None
        else:
            features['diameter'] = None  # Too expensive or disconnected
        
        return features
    
    def _calculate_suitability_scores(
        self,
        problem_type: str,
        problem_size: int,
        complexity_class: str,
        graph_features: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate suitability scores for classical and quantum solvers.
        
        Scores range from 0 (not suitable) to 1 (highly suitable).
        These scores help the router decide which solver type to use.
        
        Classical Score Factors:
        - Small problems: Very suitable (exact algorithms)
        - Polynomial complexity: High suitability
        - Dense graphs: Classical algorithms handle well
        - High clustering: Can exploit locality
        
        Quantum Score Factors:
        - Medium size (10-50): Sweet spot for quantum
        - NP-hard: Potential quantum advantage
        - Sparse graphs: Fewer quantum gates needed
        - Certain problem types (MaxCut, etc.)
        
        Args:
            problem_type: Problem type identifier
            problem_size: Number of variables
            complexity_class: Computational complexity
            graph_features: Graph analysis results (or None)
        
        Returns:
            Dictionary with 'classical_score' and 'quantum_score'
        """
        classical_score = 0.5
        quantum_score = 0.5
        
        # Size-based scoring
        if problem_size <= 20:
            # Small: classical excels
            classical_score += 0.3
            quantum_score -= 0.2
        elif problem_size <= 50:
            # Medium: both can work
            classical_score += 0.1
            quantum_score += 0.2
        else:
            # Large: depends on problem structure
            classical_score += 0.0
            quantum_score -= 0.1
        
        # Complexity class
        if 'np' in complexity_class.lower():
            quantum_score += 0.2
        if complexity_class.lower() == 'p':
            classical_score += 0.2
        
        # Problem type
        quantum_friendly = ['maxcut', 'graph_coloring', 'sat']
        if problem_type.lower() in quantum_friendly:
            quantum_score += 0.2
        else:
            classical_score += 0.1
        
        # Graph structure (if available)
        if graph_features:
            density = graph_features.get('density', 0.5)
            clustering = graph_features.get('clustering_coefficient', 0.5)
            
            # Sparse graphs favor quantum
            if density < 0.3:
                quantum_score += 0.1
            elif density > 0.7:
                classical_score += 0.1
            
            # High clustering favors classical
            if clustering > 0.5:
                classical_score += 0.1
        
        # Normalize to [0, 1]
        classical_score = max(0.0, min(1.0, classical_score))
        quantum_score = max(0.0, min(1.0, quantum_score))
        
        return {
            'classical_score': classical_score,
            'quantum_score': quantum_score
        }
    
    # =========================================================================
    # Public Utility Methods
    # =========================================================================
    
    def calculate_problem_hardness(self, problem: ProblemBase) -> str:
        """
        Calculate overall problem hardness level as a simple categorical rating.
        
        Problem hardness is a combination of multiple factors that affect how
        difficult it is to find optimal or near-optimal solutions. This method
        provides a simple 3-level categorization useful for high-level routing
        decisions and resource allocation.
        
        Hardness Factors Considered:
        -----------------------------
        
        1. **Problem Size** (Primary Factor):
           - Size determines the search space exponentially: 2^n for binary problems
           - Small problems (n≤20): Can often solve exactly
           - Medium problems (20<n≤50): Need approximations
           - Large problems (n>50): Require heuristics
        
        2. **Graph Structure Complexity** (for graph-based problems):
           - Density: Dense graphs have more interactions → harder
           - Clustering: High clustering can be exploited → easier
           - Diameter: Large diameter means information flows slowly → harder
           - Average degree: High degree means more constraints → harder
        
        3. **Known Complexity Class**:
           - P: Polynomial algorithms exist → easier
           - NP: Solution verification is polynomial, but finding is hard
           - NP-hard/NP-complete: No known polynomial solutions → harder
        
        Hardness Categories:
        -------------------
        - **'easy'**: Can likely solve to optimality quickly
          * Small size (≤20 variables)
          * OR: Polynomial complexity class
          * OR: Simple structure (low density, small diameter)
        
        - **'medium'**: Need approximation algorithms, moderate resources
          * Medium size (20-50 variables)
          * NP-hard but manageable structure
          * Can get good approximate solutions
        
        - **'hard'**: Very challenging, may not find optimal solution
          * Large size (>50 variables)
          * Dense, complex structure
          * NP-hard with no special structure to exploit
        
        Args:
            problem: Problem instance to evaluate hardness for
        
        Returns:
            Hardness level: 'easy', 'medium', or 'hard'
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=15)
            >>> problem.generate(edge_probability=0.3)
            >>> analyzer = ProblemAnalyzer()
            >>> hardness = analyzer.calculate_problem_hardness(problem)
            >>> print(f"Problem hardness: {hardness}")  # 'easy'
            >>> 
            >>> # Use for resource allocation
            >>> if hardness == 'hard':
            ...     allocate_more_time_and_resources()
        
        Mathematical Intuition:
        ----------------------
        The hardness scoring uses a point system:
        - Start with base score = 0
        - Add points for factors that increase difficulty
        - Subtract points for factors that decrease difficulty
        - Map final score to categories: <3='easy', 3-6='medium', >6='hard'
        
        This approach is interpretable and can be easily tuned based on
        empirical performance data.
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before hardness calculation")
        
        logger.debug(f"Calculating hardness for {problem}")
        
        # Initialize hardness score (higher = harder)
        hardness_score = 0
        
        size = problem.problem_size
        complexity_class = problem.complexity_class.lower()
        
        # Factor 1: Problem Size
        # This is the most important factor as search space grows exponentially
        # Small problems can often be solved exactly, large problems require heuristics
        if size <= 10:
            hardness_score += 0  # Very small, easy to solve
        elif size <= 20:
            hardness_score += 2  # Small, manageable
        elif size <= 50:
            hardness_score += 4  # Medium, need good algorithms
        elif size <= 100:
            hardness_score += 6  # Large, challenging
        else:
            hardness_score += 8  # Very large, very hard
        
        logger.debug(f"After size factor (size={size}): score={hardness_score}")
        
        # Factor 2: Complexity Class
        # Theoretical complexity gives us bounds on best possible algorithms
        # P problems have polynomial algorithms, NP-hard may not
        if complexity_class == 'p':
            hardness_score -= 2  # Efficient algorithms exist
        elif 'np-hard' in complexity_class or 'np-complete' in complexity_class:
            hardness_score += 2  # No known efficient algorithms
        elif 'np' in complexity_class:
            hardness_score += 1  # Difficult but not worst-case
        
        logger.debug(f"After complexity factor (class={complexity_class}): score={hardness_score}")
        
        # Factor 3: Graph Structure (if applicable)
        # Certain graph structures are easier to solve despite large size
        try:
            graph = problem.to_graph()
            
            # Density: More edges = more interactions = harder
            # Sparse graphs (density < 0.3) are generally easier
            # Dense graphs (density > 0.7) are harder due to many constraints
            density = nx.density(graph)
            if density < 0.2:
                hardness_score -= 1  # Very sparse, easier
            elif density > 0.7:
                hardness_score += 2  # Dense, many interactions
            
            logger.debug(f"After density factor (density={density:.3f}): score={hardness_score}")
            
            # Clustering: High clustering means local structure exists
            # Algorithms can exploit this locality for efficiency
            try:
                clustering = nx.average_clustering(graph)
                if clustering > 0.6:
                    hardness_score -= 1  # High clustering, easier to decompose
            except:
                pass
            
            # Average degree: High degree means more constraints per variable
            # More constraints generally makes problems harder
            if graph.number_of_nodes() > 0:
                avg_degree = sum(d for _, d in graph.degree()) / graph.number_of_nodes()
                if avg_degree > size * 0.5:
                    hardness_score += 1  # Very connected, harder
            
            # Diameter: Large diameter means information propagates slowly
            # Can affect convergence of iterative algorithms
            # Only compute for small graphs (expensive operation)
            if graph.number_of_nodes() <= 50:
                try:
                    if nx.is_connected(graph):
                        diameter = nx.diameter(graph)
                        if diameter > size * 0.3:
                            hardness_score += 1  # Large diameter, harder
                except:
                    pass
            
            logger.debug(f"After graph structure factors: score={hardness_score}")
        
        except Exception as e:
            logger.debug(f"Could not analyze graph structure: {e}")
        
        # Map score to categories
        # Thresholds chosen based on typical problem characteristics
        # Can be tuned based on empirical solver performance
        # Very small problems (size <= 10) are always easy regardless of structure
        if size <= 10:
            hardness = 'easy'
        elif hardness_score <= 2:
            hardness = 'easy'
        elif hardness_score <= 6:
            hardness = 'medium'
        else:
            hardness = 'hard'
        
        logger.info(f"Problem hardness: {hardness} (score={hardness_score})")
        
        return hardness
    
    def estimate_solution_space_size(self, problem: ProblemBase) -> int:
        """
        Estimate the size of the solution space (number of possible solutions).
        
        The solution space size is crucial for understanding problem difficulty.
        It represents the number of distinct solutions that need to be explored
        (in the worst case) to find the optimal solution.
        
        Mathematical Background:
        ------------------------
        
        For **Binary Optimization Problems** (MaxCut, Binary Portfolio, etc.):
        - Each variable can be 0 or 1
        - n variables → 2^n possible assignments
        - Example: 10 variables → 2^10 = 1,024 solutions
        - Example: 50 variables → 2^50 ≈ 1.1 × 10^15 solutions (quadrillion!)
        
        For **Permutation Problems** (TSP):
        - n cities can be arranged in n! (factorial) ways
        - However, TSP tours are circular, so divide by n (rotations)
        - Also symmetric (clockwise = counterclockwise), so divide by 2
        - Effective space: n! / (2n) = (n-1)! / 2
        - Example: 10 cities → 9!/2 = 181,440 tours
        - Example: 20 cities → 19!/2 ≈ 6.1 × 10^16 tours
        
        For **Continuous/Mixed Problems** (Continuous Portfolio):
        - Technically infinite solution space
        - Return -1 to indicate continuous space
        - These problems use different solution methods (convex optimization)
        
        Why This Matters:
        -----------------
        - **Exhaustive Search**: Not feasible for large spaces
          * 2^50 solutions would take millennia to enumerate
        - **Sampling**: Can only sample tiny fraction of space
          * Random sampling becomes ineffective
        - **Algorithm Selection**: Affects which algorithms work
          * Small spaces: Exhaustive or branch-and-bound
          * Large spaces: Heuristics, metaheuristics, or quantum
        - **Quantum Advantage**: Quantum algorithms can search exponentially
          large spaces more efficiently (Grover's algorithm: O(√N) vs O(N))
        
        Args:
            problem: Problem instance to estimate space size for
        
        Returns:
            Number of possible solutions (int)
            - Returns -1 for continuous problems (infinite space)
            - For very large spaces, returns sys.maxsize (overflow protection)
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=20)
            >>> problem.generate()
            >>> analyzer = ProblemAnalyzer()
            >>> space_size = analyzer.estimate_solution_space_size(problem)
            >>> print(f"Solution space size: {space_size:,}")  # 1,048,576
            >>> print(f"That's 2^{problem.problem_size}")
            >>> 
            >>> # Compare with TSP
            >>> tsp = TSPProblem(num_cities=20)
            >>> tsp.generate()
            >>> tsp_space = analyzer.estimate_solution_space_size(tsp)
            >>> print(f"TSP space: {tsp_space:,}")  # Much larger!
        
        Computational Note:
        -------------------
        For very large problems, 2^n or n! can overflow. We use arbitrary
        precision for accurate calculation up to Python's integer limits.
        For display purposes, use scientific notation for huge numbers.
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before space size estimation")
        
        problem_type = problem.problem_type.lower()
        size = problem.problem_size
        
        logger.debug(f"Estimating solution space size for {problem_type} problem of size {size}")
        
        # Binary optimization problems: 2^n solutions
        # Each of n variables can be 0 or 1, giving 2^n combinations
        if problem_type in ['maxcut', 'sat', 'graph_coloring']:
            # Calculate 2^n
            # For large n, this can be huge (2^100 ≈ 10^30)
            try:
                space_size = 2 ** size
                logger.debug(f"Binary problem: 2^{size} = {space_size}")
            except OverflowError:
                # If somehow overflow (shouldn't happen with Python's arbitrary precision)
                import sys
                space_size = sys.maxsize
                logger.warning(f"Solution space overflow, returning maxsize")
            
            return space_size
        
        # Permutation problems (TSP): (n-1)!/2 distinct tours
        # n! ways to arrange n items
        # Divide by n (circular rotations are same tour)
        # Divide by 2 (clockwise/counterclockwise are same)
        # Result: n!/(2n) = (n-1)!/2
        elif problem_type == 'tsp':
            if size <= 1:
                return 1
            
            # Calculate (n-1)! / 2
            # Use factorial from math library
            import math
            try:
                # (n-1)! can get extremely large
                # 20! = 2,432,902,008,176,640,000
                # 100! ≈ 10^157 (more atoms than in universe!)
                factorial = math.factorial(size - 1)
                space_size = factorial // 2  # Integer division
                logger.debug(f"TSP: ({size}-1)!/2 = {factorial}/2 = {space_size}")
            except (OverflowError, ValueError):
                import sys
                space_size = sys.maxsize
                logger.warning(f"TSP space size overflow for size={size}")
            
            return space_size
        
        # Portfolio optimization: depends on constraints
        # If binary (asset selection): 2^n
        # If continuous (weight allocation): infinite (return -1)
        elif problem_type == 'portfolio':
            # Check if problem has discrete or continuous variables
            # For now, assume binary (asset selection problem)
            # If it's continuous allocation, it would be indicated in problem metadata
            
            # Try to determine from problem structure
            # Most portfolio problems are continuous, so default to -1
            try:
                # If QUBO exists, it's binary
                qubo = problem.to_qubo()
                space_size = 2 ** size
                logger.debug(f"Binary portfolio: 2^{size} = {space_size}")
            except:
                # Continuous space
                space_size = -1
                logger.debug(f"Continuous portfolio: infinite solution space")
            
            return space_size
        
        # Unknown problem type: assume binary as safe default
        else:
            logger.warning(f"Unknown problem type '{problem_type}', assuming binary")
            try:
                space_size = 2 ** size
                logger.debug(f"Default binary: 2^{size} = {space_size}")
            except OverflowError:
                import sys
                space_size = sys.maxsize
            
            return space_size
    
    def identify_problem_structure(self, problem: ProblemBase) -> Dict[str, Any]:
        """
        Identify special structural properties of the problem.
        
        Certain graph structures have special properties that can be exploited
        by specialized algorithms, making problems easier to solve despite
        their theoretical complexity. This method detects these structures.
        
        Special Structures Detected:
        ----------------------------
        
        1. **Planar Graphs**:
           - Can be drawn on a plane without edge crossings
           - Many NP-hard problems become polynomial on planar graphs
           - Examples: Grid graphs, trees, outerplanar graphs
           - MaxCut on planar graphs has better approximation algorithms
           - Why it matters: Can use specialized planar graph algorithms
        
        2. **Bipartite Graphs**:
           - Nodes can be divided into two sets with edges only between sets
           - Many problems become easier (e.g., matching, vertex cover)
           - Can be checked in linear time using BFS/DFS coloring
           - MaxCut on bipartite graphs is polynomial! (not NP-hard)
           - Why it matters: Can solve optimally with efficient algorithms
        
        3. **Community Structure** (Modularity):
           - Graph has densely connected clusters with sparse inter-cluster edges
           - Real-world networks often have this structure (social, biological)
           - Can use divide-and-conquer approaches
           - Modularity score > 0.3 indicates strong communities
           - Why it matters: Can decompose problem into smaller subproblems
        
        4. **Tree Structure**:
           - Connected graph with no cycles (n nodes, n-1 edges)
           - Many NP-hard problems become polynomial on trees
           - Dynamic programming works efficiently on trees
           - Why it matters: Can use tree DP algorithms
        
        5. **Regular Graphs**:
           - All nodes have the same degree
           - Symmetric structure, easier to analyze
           - Examples: Cycle graphs, complete graphs, k-regular graphs
        
        Mathematical Details:
        ---------------------
        
        **Planarity Testing**:
        - Uses Kuratowski's theorem or Boyer-Myrvold algorithm
        - Time complexity: O(n) linear in graph size
        - NetworkX implementation: nx.check_planarity()
        
        **Bipartiteness Testing**:
        - Attempts to 2-color the graph using BFS
        - If successful → bipartite, if conflict → not bipartite
        - Time complexity: O(n + m) where m = edges
        - NetworkX implementation: nx.is_bipartite()
        
        **Community Detection** (Modularity):
        - Modularity Q = (1/2m) Σ[Aᵢⱼ - kᵢkⱼ/2m]δ(cᵢ,cⱼ)
        - Where: Aᵢⱼ = adjacency matrix, kᵢ = degree of i
        - Q > 0.3: Strong community structure
        - Q > 0.7: Very strong community structure
        - Uses Louvain algorithm (greedy optimization)
        
        Args:
            problem: Problem instance to analyze structure for
        
        Returns:
            Dictionary containing:
            {
                'is_planar': bool,          # Can be drawn without crossings
                'is_bipartite': bool,       # Two-colorable
                'is_tree': bool,            # Connected acyclic
                'is_regular': bool,         # All nodes same degree
                'has_communities': bool,    # Strong community structure
                'modularity': float or None,  # Community strength (0-1)
                'num_components': int,      # Number of connected components
                'structure_notes': List[str]  # Human-readable observations
            }
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=20)
            >>> problem.generate(edge_probability=0.2)
            >>> analyzer = ProblemAnalyzer()
            >>> structure = analyzer.identify_problem_structure(problem)
            >>> 
            >>> if structure['is_bipartite']:
            ...     print("Bipartite! Can solve optimally in polynomial time!")
            >>> if structure['has_communities']:
            ...     print(f"Has communities (modularity={structure['modularity']:.2f})")
            ...     print("Can use divide-and-conquer approach")
            >>> 
            >>> print("Structure notes:")
            >>> for note in structure['structure_notes']:
            ...     print(f"  - {note}")
        
        Performance Notes:
        ------------------
        - Planarity testing: O(n) but with high constant
        - Bipartite testing: O(n + m) very fast
        - Community detection: O(n log n) with Louvain
        - For large graphs (n > 1000), some tests may be slow
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before structure identification")
        
        logger.debug(f"Identifying structural properties of {problem}")
        
        # Initialize results dictionary
        structure = {
            'is_planar': False,
            'is_bipartite': False,
            'is_tree': False,
            'is_regular': False,
            'has_communities': False,
            'modularity': None,
            'num_components': 0,
            'structure_notes': []
        }
        
        # Try to get graph representation
        # Not all problems have meaningful graph representations
        try:
            graph = problem.to_graph()
        except Exception as e:
            logger.warning(f"Could not get graph representation: {e}")
            structure['structure_notes'].append("Not a graph-based problem")
            return structure
        
        # Basic graph properties
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        if n == 0:
            structure['structure_notes'].append("Empty graph")
            return structure
        
        logger.debug(f"Graph has {n} nodes and {m} edges")
        
        # Check 1: Connected Components
        # Number of disconnected parts of the graph
        # Can solve each component independently (divide and conquer!)
        if graph.is_directed():
            num_components = nx.number_weakly_connected_components(graph)
        else:
            num_components = nx.number_connected_components(graph)
        
        structure['num_components'] = num_components
        
        if num_components > 1:
            structure['structure_notes'].append(
                f"Graph has {num_components} components - can solve independently"
            )
            logger.debug(f"Graph is disconnected: {num_components} components")
        
        # For remaining tests, work with largest component if disconnected
        if num_components > 1 and not graph.is_directed():
            # Get largest connected component for further analysis
            largest_cc = max(nx.connected_components(graph), key=len)
            graph_to_analyze = graph.subgraph(largest_cc).copy()
            logger.debug(f"Using largest component with {graph_to_analyze.number_of_nodes()} nodes")
        else:
            graph_to_analyze = graph
        
        # Check 2: Bipartiteness
        # Can we 2-color the graph? (nodes in two sets, edges only between sets)
        # IMPORTANT: MaxCut on bipartite graphs is in P (polynomial time)!
        try:
            is_bipartite = nx.is_bipartite(graph_to_analyze)
            structure['is_bipartite'] = is_bipartite
            
            if is_bipartite:
                structure['structure_notes'].append(
                    "Bipartite structure detected - many NP-hard problems become polynomial!"
                )
                logger.info("Graph is BIPARTITE - special algorithms available")
        except Exception as e:
            logger.debug(f"Could not check bipartiteness: {e}")
        
        # Check 3: Tree Structure
        # Trees are connected acyclic graphs: n nodes, n-1 edges
        # Many NP-hard problems become polynomial on trees (tree DP)
        if num_components == 1:
            is_tree = nx.is_tree(graph_to_analyze)
            structure['is_tree'] = is_tree
            
            if is_tree:
                structure['structure_notes'].append(
                    "Tree structure - can use dynamic programming on trees"
                )
                logger.info("Graph is a TREE - many efficient algorithms available")
        
        # Check 4: Planarity
        # Can graph be drawn on plane without edge crossings?
        # Many problems have better algorithms on planar graphs
        # Note: Expensive for large graphs, so limit size
        if n <= 1000:  # Planarity testing can be slow for very large graphs
            try:
                is_planar, _ = nx.check_planarity(graph_to_analyze)
                structure['is_planar'] = is_planar
                
                if is_planar:
                    structure['structure_notes'].append(
                        "Planar graph - specialized planar algorithms available"
                    )
                    logger.info("Graph is PLANAR")
            except Exception as e:
                logger.debug(f"Could not check planarity: {e}")
        else:
            logger.debug(f"Skipping planarity check for large graph (n={n})")
        
        # Check 5: Regularity
        # Are all nodes of the same degree? (symmetric structure)
        # Regular graphs have nice mathematical properties
        try:
            degrees = [d for _, d in graph_to_analyze.degree()]
            if len(degrees) > 0 and len(set(degrees)) == 1:
                structure['is_regular'] = True
                structure['structure_notes'].append(
                    f"Regular graph (all nodes have degree {degrees[0]})"
                )
                logger.info(f"Graph is {degrees[0]}-REGULAR")
        except Exception as e:
            logger.debug(f"Could not check regularity: {e}")
        
        # Check 6: Community Structure (Modularity)
        # Does graph have densely connected clusters?
        # High modularity means we can decompose problem
        # Use Louvain community detection algorithm
        if n >= 10 and n <= 5000:  # Only for reasonable sizes
            try:
                # Import community detection (requires python-louvain or networkx communities)
                # Try to use greedy modularity communities (built into NetworkX)
                from networkx.algorithms import community
                
                # Find communities using greedy modularity maximization
                communities_generator = community.greedy_modularity_communities(
                    graph_to_analyze
                )
                communities_list = list(communities_generator)
                
                # Calculate modularity score
                modularity = community.modularity(graph_to_analyze, communities_list)
                structure['modularity'] = modularity
                
                # Modularity interpretation:
                # Q < 0.3: Weak or no community structure
                # 0.3 ≤ Q < 0.5: Moderate community structure
                # Q ≥ 0.5: Strong community structure
                # Q > 0.7: Very strong community structure (rare)
                if modularity > 0.3:
                    structure['has_communities'] = True
                    structure['structure_notes'].append(
                        f"Community structure detected (modularity={modularity:.3f}, "
                        f"{len(communities_list)} communities) - can use divide-and-conquer"
                    )
                    logger.info(f"Strong community structure: modularity={modularity:.3f}")
                else:
                    structure['structure_notes'].append(
                        f"Weak community structure (modularity={modularity:.3f})"
                    )
                
            except Exception as e:
                logger.debug(f"Could not detect communities: {e}")
                structure['structure_notes'].append("Community detection not available")
        else:
            logger.debug(f"Skipping community detection (n={n} outside range [10, 5000])")
        
        # Add general observations
        density = nx.density(graph_to_analyze)
        if density < 0.1:
            structure['structure_notes'].append("Very sparse graph (density < 0.1)")
        elif density > 0.8:
            structure['structure_notes'].append("Very dense graph (density > 0.8)")
        
        logger.info(f"Structure analysis complete: {len(structure['structure_notes'])} observations")
        
        return structure
    
    def generate_analysis_report(self, problem: ProblemBase) -> str:
        """
        Generate a comprehensive human-readable analysis report.
        
        This method combines all analysis functions to create a detailed
        report suitable for:
        - Understanding problem characteristics
        - Making routing decisions
        - Debugging and development
        - User feedback and transparency
        
        The report includes:
        - Basic problem information
        - Size and complexity analysis
        - Graph structure analysis (if applicable)
        - Runtime estimates
        - Solver recommendations
        - Special structural properties
        
        Report Format:
        --------------
        The report is formatted as plain text with sections:
        1. Problem Overview
        2. Problem Characteristics
        3. Graph Analysis (if applicable)
        4. Solution Space Analysis
        5. Computational Estimates
        6. Structural Properties
        7. Recommendations
        
        Args:
            problem: Problem instance to generate report for
        
        Returns:
            Formatted analysis report as string
        
        Raises:
            ValueError: If problem is not generated
        
        Example:
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate(edge_probability=0.3)
            >>> analyzer = ProblemAnalyzer()
            >>> report = analyzer.generate_analysis_report(problem)
            >>> print(report)
            
            ================================================================================
            PROBLEM ANALYSIS REPORT
            ================================================================================
            
            PROBLEM OVERVIEW
            ----------------
            Problem Type: maxcut
            Problem Size: 30 variables
            Complexity Class: NP-hard
            ...
        
        Use Cases:
        ----------
        1. **Debugging**: Understand why router made certain decision
        2. **User Interface**: Show analysis to users
        3. **Logging**: Record problem characteristics for later analysis
        4. **Research**: Document problem instances in papers
        """
        if not problem.is_generated:
            raise ValueError("Problem must be generated before report generation")
        
        logger.info(f"Generating analysis report for {problem}")
        
        # Collect all analysis data
        analysis = self.analyze_problem(problem)
        hardness = self.calculate_problem_hardness(problem)
        space_size = self.estimate_solution_space_size(problem)
        structure = self.identify_problem_structure(problem)
        
        # Build report sections
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("PROBLEM ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Section 1: Problem Overview
        report_lines.append("PROBLEM OVERVIEW")
        report_lines.append("-" * 80)
        report_lines.append(f"Problem Type: {analysis['problem_type']}")
        report_lines.append(f"Problem Size: {analysis['problem_size']} variables")
        report_lines.append(f"Complexity Class: {problem.complexity_class}")
        report_lines.append(f"Problem Hardness: {hardness.upper()}")
        report_lines.append("")
        
        # Section 2: Solution Space
        report_lines.append("SOLUTION SPACE ANALYSIS")
        report_lines.append("-" * 80)
        if space_size == -1:
            report_lines.append("Solution Space: Continuous (infinite)")
        else:
            # Format large numbers with scientific notation if needed
            if space_size > 1e15:
                report_lines.append(f"Solution Space Size: {space_size:.2e} possible solutions")
            else:
                report_lines.append(f"Solution Space Size: {space_size:,} possible solutions")
            
            # Add intuitive explanation
            if space_size < 1e6:
                report_lines.append("  → Small space: Exhaustive search may be feasible")
            elif space_size < 1e12:
                report_lines.append("  → Medium space: Need smart search strategies")
            else:
                report_lines.append("  → Huge space: Only tiny fraction can be explored")
        report_lines.append("")
        
        # Section 3: Graph Analysis (if available)
        if analysis['graph_features']:
            report_lines.append("GRAPH STRUCTURE ANALYSIS")
            report_lines.append("-" * 80)
            gf = analysis['graph_features']
            report_lines.append(f"Graph Density: {gf['density']:.3f}")
            report_lines.append(f"Clustering Coefficient: {gf['clustering_coefficient']:.3f}")
            report_lines.append(f"Average Degree: {gf['average_degree']:.2f}")
            if gf['diameter'] is not None:
                report_lines.append(f"Graph Diameter: {gf['diameter']}")
            else:
                report_lines.append("Graph Diameter: Not computed (too expensive or disconnected)")
            report_lines.append("")
        
        # Section 4: Structural Properties
        report_lines.append("STRUCTURAL PROPERTIES")
        report_lines.append("-" * 80)
        report_lines.append(f"Connected Components: {structure['num_components']}")
        report_lines.append(f"Bipartite: {'Yes' if structure['is_bipartite'] else 'No'}")
        report_lines.append(f"Tree Structure: {'Yes' if structure['is_tree'] else 'No'}")
        report_lines.append(f"Planar: {'Yes' if structure['is_planar'] else 'No'}")
        report_lines.append(f"Regular: {'Yes' if structure['is_regular'] else 'No'}")
        if structure['modularity'] is not None:
            report_lines.append(f"Community Structure: {'Yes' if structure['has_communities'] else 'No'} "
                              f"(modularity={structure['modularity']:.3f})")
        
        if structure['structure_notes']:
            report_lines.append("")
            report_lines.append("Structure Notes:")
            for note in structure['structure_notes']:
                report_lines.append(f"  • {note}")
        report_lines.append("")
        
        # Section 5: Computational Estimates
        report_lines.append("COMPUTATIONAL ESTIMATES")
        report_lines.append("-" * 80)
        report_lines.append(f"Estimated Classical Runtime: {analysis['estimated_classical_runtime']:.4f} seconds")
        report_lines.append(f"Estimated Quantum Runtime: {analysis['estimated_quantum_runtime']:.4f} seconds")
        
        speedup_factor = (analysis['estimated_classical_runtime'] / 
                         analysis['estimated_quantum_runtime'] 
                         if analysis['estimated_quantum_runtime'] > 0 else float('inf'))
        report_lines.append(f"Speedup Factor: {speedup_factor:.2f}x")
        report_lines.append("")
        
        # Section 6: Solver Suitability
        report_lines.append("SOLVER SUITABILITY SCORES")
        report_lines.append("-" * 80)
        classical_score = analysis['suitability_scores']['classical_score']
        quantum_score = analysis['suitability_scores']['quantum_score']
        
        report_lines.append(f"Classical Solver Score: {classical_score:.2f} / 1.00")
        report_lines.append(f"Quantum Solver Score: {quantum_score:.2f} / 1.00")
        report_lines.append(f"Quantum Advantage Probability: {analysis['quantum_advantage_probability']:.1%}")
        report_lines.append("")
        
        # Section 7: Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        # Determine recommendation based on scores and analysis
        if structure['is_bipartite'] and analysis['problem_type'] == 'maxcut':
            report_lines.append("✓ RECOMMENDED: Classical Solver (Bipartite MaxCut is polynomial)")
            report_lines.append("  Reasoning: Bipartite MaxCut can be solved optimally in polynomial time.")
        elif structure['is_tree']:
            report_lines.append("✓ RECOMMENDED: Classical Solver (Tree DP is efficient)")
            report_lines.append("  Reasoning: Many problems on trees have efficient dynamic programming solutions.")
        elif quantum_score > classical_score + 0.2:
            report_lines.append("✓ RECOMMENDED: Quantum Solver")
            report_lines.append(f"  Reasoning: Quantum score ({quantum_score:.2f}) significantly higher than "
                              f"classical ({classical_score:.2f}).")
            report_lines.append(f"  Quantum advantage probability: {analysis['quantum_advantage_probability']:.1%}")
        elif classical_score > quantum_score + 0.2:
            report_lines.append("✓ RECOMMENDED: Classical Solver")
            report_lines.append(f"  Reasoning: Classical score ({classical_score:.2f}) significantly higher than "
                              f"quantum ({quantum_score:.2f}).")
        else:
            report_lines.append("✓ RECOMMENDED: Hybrid Approach")
            report_lines.append(f"  Reasoning: Scores are close (classical={classical_score:.2f}, "
                              f"quantum={quantum_score:.2f}).")
            report_lines.append("  Consider running both solvers in parallel or using quantum for initial solution.")
        
        # Add specific notes based on hardness
        if hardness == 'hard':
            report_lines.append("")
            report_lines.append("⚠ WARNING: This is a HARD problem instance.")
            report_lines.append("  • May require significant computational resources")
            report_lines.append("  • Consider using heuristics or approximation algorithms")
            report_lines.append("  • Optimal solution may not be found in reasonable time")
        elif hardness == 'easy':
            report_lines.append("")
            report_lines.append("✓ NOTE: This is an EASY problem instance.")
            report_lines.append("  • Can likely solve to optimality quickly")
            report_lines.append("  • Exact algorithms should work well")
        
        # Footer
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Join all lines
        report = "\n".join(report_lines)
        
        logger.info("Analysis report generated successfully")
        
        return report
