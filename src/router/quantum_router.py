"""
Quantum Router for Intelligent Solver Selection

This module implements intelligent routing between classical and quantum solvers
based on problem characteristics, resource constraints, and performance predictions.
The router makes data-driven decisions to optimize execution time, energy efficiency,
and solution quality.

Key Features:
- Multi-factor decision making (problem analysis, resources, performance)
- Detailed reasoning and confidence scoring
- Alternative routing suggestions
- Energy efficiency optimization
- Support for hybrid approaches

The router integrates:
- ProblemAnalyzer: For problem characteristic extraction
- EdgeEnvironment: For resource constraint validation
- Historical data: For performance prediction (future enhancement)
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import statistics

from src.analyzer.problem_analyzer import ProblemAnalyzer
from src.router.edge_simulator import EdgeEnvironment, JobRequirements
from src.problems.problem_base import ProblemBase


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class JobExecution:
    """
    Record of a completed job execution for historical learning.
    
    This dataclass stores both predicted and actual performance metrics,
    enabling the router to learn from experience and improve predictions
    over time. The gap between predictions and reality helps calibrate
    future routing decisions.
    
    Attributes:
        job_id: Unique identifier for the job
        timestamp: When the job was executed
        problem_type: Type of problem (maxcut, tsp, portfolio, etc.)
        problem_size: Number of variables/nodes
        solver_used: Which solver was chosen (classical, quantum, hybrid)
        
        # Predictions (from routing decision)
        estimated_time_s: Predicted execution time
        estimated_energy_mj: Predicted energy consumption
        predicted_quantum_advantage: Quantum advantage probability
        
        # Actuals (from execution monitoring)
        actual_time_s: Measured execution time
        actual_energy_mj: Measured energy consumption
        solution_quality: Quality metric (depends on problem type)
        
        # Context
        edge_profile: Deployment profile used (aerospace, mobile, ground_server)
        strategy_used: Routing strategy that made decision
        
        # Outcome
        success: Whether job completed successfully
        error_message: Error details if failed
    """
    job_id: str
    timestamp: datetime
    problem_type: str
    problem_size: int
    solver_used: str
    
    # Predictions
    estimated_time_s: float
    estimated_energy_mj: float
    predicted_quantum_advantage: float
    
    # Actuals
    actual_time_s: Optional[float] = None
    actual_energy_mj: Optional[float] = None
    solution_quality: Optional[float] = None
    
    # Context
    edge_profile: Optional[str] = None
    strategy_used: Optional[str] = None
    
    # Outcome
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PerformanceStatistics:
    """
    Aggregated performance statistics for a category of jobs.
    
    This dataclass holds statistics computed from historical executions,
    used to improve routing decisions for similar future jobs.
    
    Attributes:
        problem_type: Problem type these stats apply to
        size_range: (min, max) problem size range
        solver_type: Solver these stats are for (classical or quantum)
        
        num_executions: Number of jobs in this category
        
        avg_time_s: Average execution time
        std_time_s: Standard deviation of execution time
        avg_energy_mj: Average energy consumption
        std_energy_mj: Standard deviation of energy
        avg_quality: Average solution quality
        
        success_rate: Fraction of jobs that succeeded
        avg_quantum_advantage: Average observed quantum advantage
        avg_prediction_error_time: Average |actual - predicted| time
        avg_prediction_error_energy: Average |actual - predicted| energy
    """
    problem_type: str
    size_range: Tuple[int, int]
    solver_type: str
    
    num_executions: int = 0
    
    avg_time_s: float = 0.0
    std_time_s: float = 0.0
    avg_energy_mj: float = 0.0
    std_energy_mj: float = 0.0
    avg_quality: float = 0.0
    
    success_rate: float = 0.0
    avg_quantum_advantage: float = 0.0
    avg_prediction_error_time: float = 0.0
    avg_prediction_error_energy: float = 0.0


class RoutingStrategy(Enum):
    """
    Advanced routing strategies for different optimization objectives.
    
    Each strategy prioritizes different aspects of execution:
    - ENERGY_OPTIMIZED: Minimize total energy consumption (best for battery-powered)
    - LATENCY_OPTIMIZED: Minimize execution time (best for real-time applications)
    - QUALITY_OPTIMIZED: Maximize solution quality (best for critical decisions)
    - BALANCED: Balance time, energy, and quality (general purpose)
    
    Strategies affect how the router scores classical vs quantum options
    and can override default decision tree logic when appropriate.
    """
    ENERGY_OPTIMIZED = "energy_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"


@dataclass
class RoutingPreferences:
    """
    User preferences for routing decisions.
    
    Allows customization of routing behavior based on use case requirements.
    Different applications have different priorities (speed vs quality vs energy).
    """
    prefer_speed: bool = True          # Minimize execution time
    prefer_quality: bool = False       # Maximize solution quality
    prefer_energy: bool = False        # Minimize energy consumption
    allow_hybrid: bool = True          # Allow hybrid quantum-classical approaches
    min_confidence: float = 0.5        # Minimum confidence threshold for quantum routing
    max_runtime_seconds: float = 60.0  # Maximum acceptable runtime


class QuantumRouter:
    """
    Intelligent routing system for quantum-classical hybrid computing.
    
    The QuantumRouter makes intelligent decisions about whether to execute
    problems on classical or quantum solvers, based on comprehensive analysis
    of problem characteristics, resource constraints, and performance predictions.
    
    Decision Process:
    -----------------
    1. **Problem Analysis**: Extract features (size, structure, complexity)
    2. **Resource Validation**: Check if resources available for each solver
    3. **Performance Prediction**: Estimate runtime, energy, solution quality
    4. **Decision Logic**: Apply decision tree with multiple criteria
    5. **Confidence Scoring**: Quantify certainty of routing decision
    
    Why Intelligent Routing Matters:
    --------------------------------
    
    **For Classical Solvers**:
    - Very efficient for small problems (< 10 variables)
    - Mature algorithms with predictable performance
    - Low resource overhead
    - BUT: Exponential scaling limits problem size
    
    **For Quantum Solvers (Simulators)**:
    - Potential advantage for medium-size problems (10-50 variables)
    - Can explore exponentially large solution spaces
    - Rotonium's room-temp QPU enables edge deployment
    - BUT: Simulation overhead, qubit limitations, requires more resources
    
    **For Hybrid Approaches**:
    - Use quantum for initial solution, classical for refinement
    - Decompose problem: quantum for hard parts, classical for easy parts
    - Iterative: quantum optimization with classical post-processing
    
    The router balances these tradeoffs based on:
    - Problem characteristics (detected by ProblemAnalyzer)
    - Available resources (from EdgeEnvironment)
    - User preferences (speed, quality, energy efficiency)
    - Predicted performance (runtime, energy, solution quality)
    
    Rotonium Technology Advantages:
    -------------------------------
    Traditional quantum computers need cryogenic cooling (1000+ watts),
    making them impractical for edge deployment. Rotonium's room-temperature
    molecular quantum processor:
    - Operates at ambient temperature (no cryogenics!)
    - Consumes < 50 watts (battery-friendly)
    - Compact form factor (fits in edge devices)
    - Enables aerospace, mobile, and ground server quantum computing
    
    This router optimizes for Rotonium's unique characteristics, enabling
    practical quantum computing in resource-constrained environments.
    
    Attributes:
        analyzer: ProblemAnalyzer instance for problem feature extraction
        current_strategy: Active routing strategy (default: BALANCED)
        execution_history: List of completed job executions for learning
        performance_cache: Cached statistics by (problem_type, size_range, solver)
        learning_enabled: Whether to learn from historical executions
        routing_thresholds: Dynamically adjusted decision thresholds
    """
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.BALANCED,
                 enable_learning: bool = True):
        """
        Initialize the quantum router with a problem analyzer.
        
        Args:
            strategy: Initial routing strategy (default: BALANCED)
            enable_learning: Enable learning from historical executions (default: True)
        """
        self.analyzer = ProblemAnalyzer()
        self.current_strategy = strategy
        
        # Historical performance tracking
        self.learning_enabled = enable_learning
        self.execution_history: List[JobExecution] = []
        self.performance_cache: Dict[Tuple[str, Tuple[int, int], str], PerformanceStatistics] = {}
        
        # Dynamic routing thresholds (can be adjusted based on history)
        # These override default thresholds when sufficient data exists
        self.routing_thresholds = {
            'min_problem_size': 10,     # Below this, always use classical
            'max_problem_size': 100,    # Above this, classical more practical
            'quantum_advantage_threshold': 0.6,  # Min advantage for quantum routing
            'confidence_threshold': 0.5,  # Min confidence to trust quantum
        }
        
        logger.info(f"QuantumRouter initialized with strategy: {strategy.value}, "
                   f"learning_enabled: {enable_learning}")
        self.current_strategy = strategy
        logger.info(f"QuantumRouter initialized with strategy: {strategy.value}")
    
    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """
        Configure routing strategy to optimize for specific objectives.
        
        This method allows dynamic reconfiguration of routing priorities
        without creating a new router instance. Useful for applications
        that need to switch strategies based on context:
        - Battery low → ENERGY_OPTIMIZED
        - Critical mission phase → QUALITY_OPTIMIZED
        - Real-time navigation → LATENCY_OPTIMIZED
        - Normal operations → BALANCED
        
        Strategy Characteristics:
        -------------------------
        
        **ENERGY_OPTIMIZED**:
        - Prioritizes minimizing total energy consumption
        - Prefers quantum if energy/solution ratio is lower
        - Best for: Battery-powered edge devices (aerospace, mobile)
        - Trade-off: May accept slightly longer runtime for energy savings
        - Rotonium advantage: Room-temp QPU often more efficient than classical
        
        **LATENCY_OPTIMIZED**:
        - Prioritizes minimizing execution time
        - May use more power to finish faster
        - Best for: Real-time applications (navigation, sensing, control)
        - Trade-off: Higher energy consumption for faster results
        - Considers: Circuit depth, compilation overhead, measurement time
        
        **QUALITY_OPTIMIZED**:
        - Prioritizes best solution quality
        - Willing to spend more time and energy for better results
        - Best for: Critical decisions (mission planning, resource allocation)
        - Trade-off: Higher resource consumption for solution optimality
        - May prefer: Quantum for better exploration of solution space
        
        **BALANCED**:
        - Balances time, energy, and quality using weighted scoring
        - General-purpose strategy for typical use cases
        - Weights: 40% time, 30% energy, 30% quality
        - Best for: Mixed workloads, normal operations
        - Adapts: To problem characteristics and resource availability
        
        Args:
            strategy: New routing strategy to use
        
        Example:
            >>> router = QuantumRouter()
            >>> 
            >>> # Normal operations
            >>> router.set_strategy(RoutingStrategy.BALANCED)
            >>> result1 = router.route_problem(problem1, env)
            >>> 
            >>> # Battery running low
            >>> if battery_level < 20:
            ...     router.set_strategy(RoutingStrategy.ENERGY_OPTIMIZED)
            >>> 
            >>> # Critical mission phase
            >>> if mission_phase == "critical":
            ...     router.set_strategy(RoutingStrategy.QUALITY_OPTIMIZED)
            >>> 
            >>> # Real-time navigation
            >>> if mode == "navigation":
            ...     router.set_strategy(RoutingStrategy.LATENCY_OPTIMIZED)
        
        Notes:
            - Strategy change takes effect immediately for next routing
            - Does not affect already-routed problems
            - Strategy is preserved across multiple route_problem() calls
            - Can be changed dynamically based on system state
        """
        old_strategy = self.current_strategy
        self.current_strategy = strategy
        logger.info(f"Routing strategy changed from {old_strategy.value} to {strategy.value}")
    
    def calculate_strategy_score(
        self,
        classical_est: Dict[str, float],
        quantum_est: Dict[str, float],
        strategy: Optional[RoutingStrategy] = None
    ) -> Tuple[str, float, str]:
        """
        Score classical and quantum options based on routing strategy.
        
        This method quantifies how well each solver option aligns with
        the current strategy's objectives. Higher scores indicate better
        alignment with strategy goals.
        
        Scoring Logic by Strategy:
        ---------------------------
        
        **ENERGY_OPTIMIZED**:
        - Score = 1000 / (energy_mj + 1)
        - Lower energy → Higher score
        - Additional bonus: If energy < 50% of alternative, +20% score
        - Rationale: Direct optimization of energy consumption
        - Units: Inverse energy (higher = more efficient)
        
        **LATENCY_OPTIMIZED**:
        - Score = 1000 / (runtime_s + 0.1)
        - Lower runtime → Higher score
        - Additional bonus: If runtime < 70% of alternative, +15% score
        - Rationale: Direct optimization of execution time
        - Units: Inverse time (higher = faster)
        
        **QUALITY_OPTIMIZED**:
        - Base score = quantum_advantage_probability × 100
        - Quantum: score × (1 + 0.3) if advantage > 0.6
        - Classical: score × (1 + 0.2) if advantage < 0.4
        - Rationale: Prefer solver likely to give better solution
        - Units: Expected quality improvement (0-100+ scale)
        
        **BALANCED**:
        - Weighted combination of all factors
        - Time score (40%): 1000 / (runtime_s + 0.1)
        - Energy score (30%): 1000 / (energy_mj + 1)
        - Quality score (30%): quantum_advantage_prob × 100
        - Normalize and combine: 0.4×time + 0.3×energy + 0.3×quality
        - Rationale: No single factor dominates decision
        - Units: Composite score (0-100 scale)
        
        Score Interpretation:
        ---------------------
        - Scores are relative (compare classical vs quantum)
        - Higher score = better alignment with strategy
        - Score difference > 10%: Clear preference
        - Score difference < 5%: Toss-up, use confidence factors
        - Absolute score values less important than relative comparison
        
        Args:
            classical_est: Classical solver estimates
                {
                    'runtime_s': float,
                    'energy_mj': float,
                    'fits_resources': bool
                }
            quantum_est: Quantum solver estimates (same structure)
            strategy: Strategy to use (default: self.current_strategy)
        
        Returns:
            Tuple of (decision, score_diff, reasoning):
            - decision: 'classical' or 'quantum'
            - score_diff: Score difference (positive = quantum better)
            - reasoning: Human-readable explanation
        
        Example:
            >>> classical_est = {
            ...     'runtime_s': 5.0,
            ...     'energy_mj': 150.0,
            ...     'fits_resources': True
            ... }
            >>> quantum_est = {
            ...     'runtime_s': 3.0,
            ...     'energy_mj': 120.0,
            ...     'fits_resources': True
            ... }
            >>> 
            >>> # Energy-optimized: quantum uses less energy
            >>> decision, diff, reason = router.calculate_strategy_score(
            ...     classical_est, quantum_est, RoutingStrategy.ENERGY_OPTIMIZED
            ... )
            >>> print(f"{decision}: {reason}")
            quantum: Quantum solver uses 20% less energy (120.0mJ vs 150.0mJ)
        
        Notes:
            - Both options must fit resources to be considered
            - If only one fits, it wins by default (infinite score diff)
            - Quality estimates based on quantum advantage probability
            - Scoring is deterministic (same inputs → same scores)
        """
        
        # Use current strategy if not specified
        if strategy is None:
            strategy = self.current_strategy
        
        logger.debug(f"Calculating scores for strategy: {strategy.value}")
        
        # Extract estimates
        classical_runtime = classical_est.get('runtime_s', 1.0)
        classical_energy = classical_est.get('energy_mj', 100.0)
        classical_fits = classical_est.get('fits_resources', True)
        
        quantum_runtime = quantum_est.get('runtime_s', 1.0)
        quantum_energy = quantum_est.get('energy_mj', 100.0)
        quantum_fits = quantum_est.get('fits_resources', True)
        quantum_advantage = quantum_est.get('quantum_advantage_prob', 0.5)
        
        # Handle resource constraint cases first
        # If only one fits, it wins automatically
        if classical_fits and not quantum_fits:
            return ('classical', 1000.0, 
                   'Classical solver is only option that fits resource constraints')
        
        if quantum_fits and not classical_fits:
            return ('quantum', 1000.0,
                   'Quantum solver is only option that fits resource constraints')
        
        if not classical_fits and not quantum_fits:
            # Both don't fit - should not happen, but handle gracefully
            return ('classical', 0.0,
                   'Neither solver fits constraints, defaulting to classical')
        
        # Both fit - score based on strategy
        
        if strategy == RoutingStrategy.ENERGY_OPTIMIZED:
            # ================================================================
            # ENERGY_OPTIMIZED: Minimize total energy consumption
            # ================================================================
            # Score = 1000 / (energy + 1)
            # Lower energy = higher score
            # Bonus if energy is significantly lower than alternative
            
            classical_score = 1000.0 / (classical_energy + 1.0)
            quantum_score = 1000.0 / (quantum_energy + 1.0)
            
            # Apply bonus for significant energy savings (>50% reduction)
            if quantum_energy < classical_energy * 0.5:
                quantum_score *= 1.2  # 20% bonus
            elif classical_energy < quantum_energy * 0.5:
                classical_score *= 1.2
            
            score_diff = quantum_score - classical_score
            
            if score_diff > 0.5:
                decision = 'quantum'
                energy_savings = classical_energy - quantum_energy
                pct_savings = (energy_savings / classical_energy) * 100
                reasoning = (
                    f"ENERGY_OPTIMIZED strategy: Quantum solver uses {pct_savings:.1f}% "
                    f"less energy ({quantum_energy:.1f}mJ vs {classical_energy:.1f}mJ). "
                    f"Critical for battery-powered edge devices. Rotonium's room-temp QPU "
                    f"advantage is maximized in energy-constrained scenarios."
                )
            elif score_diff < -0.5:
                decision = 'classical'
                energy_savings = quantum_energy - classical_energy
                pct_savings = (energy_savings / quantum_energy) * 100
                reasoning = (
                    f"ENERGY_OPTIMIZED strategy: Classical solver uses {pct_savings:.1f}% "
                    f"less energy ({classical_energy:.1f}mJ vs {quantum_energy:.1f}mJ). "
                    f"For this problem size, classical processing is more energy-efficient."
                )
            else:
                decision = 'quantum' if quantum_score >= classical_score else 'classical'
                reasoning = (
                    f"ENERGY_OPTIMIZED strategy: Energy consumption similar "
                    f"(classical={classical_energy:.1f}mJ, quantum={quantum_energy:.1f}mJ). "
                    f"Selecting {decision} based on slight efficiency advantage."
                )
        
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            # ================================================================
            # LATENCY_OPTIMIZED: Minimize execution time
            # ================================================================
            # Score = 1000 / (runtime + 0.1)
            # Lower runtime = higher score
            # Bonus if runtime is significantly lower than alternative
            
            classical_score = 1000.0 / (classical_runtime + 0.1)
            quantum_score = 1000.0 / (quantum_runtime + 0.1)
            
            # Apply bonus for significant speedup (>30% faster)
            if quantum_runtime < classical_runtime * 0.7:
                quantum_score *= 1.15  # 15% bonus
            elif classical_runtime < quantum_runtime * 0.7:
                classical_score *= 1.15
            
            score_diff = quantum_score - classical_score
            
            if score_diff > 0.5:
                decision = 'quantum'
                speedup = classical_runtime / quantum_runtime
                time_saved = classical_runtime - quantum_runtime
                reasoning = (
                    f"LATENCY_OPTIMIZED strategy: Quantum solver is {speedup:.2f}x faster "
                    f"({quantum_runtime:.3f}s vs {classical_runtime:.3f}s), saving {time_saved:.3f}s. "
                    f"Critical for real-time applications (navigation, sensing, control). "
                    f"Quantum circuit execution can be faster than classical heuristics."
                )
            elif score_diff < -0.5:
                decision = 'classical'
                speedup = quantum_runtime / classical_runtime
                time_saved = quantum_runtime - classical_runtime
                reasoning = (
                    f"LATENCY_OPTIMIZED strategy: Classical solver is {speedup:.2f}x faster "
                    f"({classical_runtime:.3f}s vs {quantum_runtime:.3f}s), saving {time_saved:.3f}s. "
                    f"For this problem size, quantum overhead (compilation, measurement) "
                    f"outweighs potential speedup."
                )
            else:
                decision = 'quantum' if quantum_score >= classical_score else 'classical'
                reasoning = (
                    f"LATENCY_OPTIMIZED strategy: Runtime similar "
                    f"(classical={classical_runtime:.3f}s, quantum={quantum_runtime:.3f}s). "
                    f"Selecting {decision} based on slight speed advantage."
                )
        
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            # ================================================================
            # QUALITY_OPTIMIZED: Maximize solution quality
            # ================================================================
            # Score based on quantum advantage probability
            # Quantum gets bonus if high advantage probability (>0.6)
            # Classical gets bonus if low advantage probability (<0.4)
            # Willing to spend more resources for better solution
            
            # Base scores from quantum advantage probability
            # High probability → quantum likely gives better solution
            # Low probability → classical algorithms work well
            quantum_score = quantum_advantage * 100
            classical_score = (1.0 - quantum_advantage) * 100
            
            # Apply quality bonuses
            if quantum_advantage > 0.6:
                quantum_score *= 1.3  # 30% bonus for high quantum advantage
            elif quantum_advantage < 0.4:
                classical_score *= 1.2  # 20% bonus for low quantum advantage
            
            score_diff = quantum_score - classical_score
            
            if score_diff > 5.0:
                decision = 'quantum'
                reasoning = (
                    f"QUALITY_OPTIMIZED strategy: Quantum solver predicted to give better "
                    f"solution quality (advantage probability={quantum_advantage:.1%}). "
                    f"Quantum algorithms can explore solution space more effectively. "
                    f"Willing to accept higher resource consumption (runtime={quantum_runtime:.3f}s, "
                    f"energy={quantum_energy:.1f}mJ) for superior solution."
                )
            elif score_diff < -5.0:
                decision = 'classical'
                reasoning = (
                    f"QUALITY_OPTIMIZED strategy: Classical solver predicted to give comparable "
                    f"or better solution quality (quantum advantage probability={quantum_advantage:.1%}). "
                    f"Mature classical algorithms excel for this problem structure. "
                    f"Resource efficient (runtime={classical_runtime:.3f}s, energy={classical_energy:.1f}mJ)."
                )
            else:
                decision = 'quantum' if quantum_score >= classical_score else 'classical'
                reasoning = (
                    f"QUALITY_OPTIMIZED strategy: Solution quality expected to be similar "
                    f"(quantum advantage probability={quantum_advantage:.1%}). "
                    f"Selecting {decision} based on slight quality advantage prediction."
                )
        
        else:  # RoutingStrategy.BALANCED
            # ================================================================
            # BALANCED: Balance time, energy, and quality
            # ================================================================
            # Weighted combination: 40% time, 30% energy, 30% quality
            # Normalize each component to 0-100 scale for fair comparison
            
            # Time component (40% weight): Inverse of runtime
            time_classical = 1000.0 / (classical_runtime + 0.1)
            time_quantum = 1000.0 / (quantum_runtime + 0.1)
            # Normalize to 0-100 scale
            time_max = max(time_classical, time_quantum)
            time_classical_norm = (time_classical / time_max) * 100
            time_quantum_norm = (time_quantum / time_max) * 100
            
            # Energy component (30% weight): Inverse of energy
            energy_classical = 1000.0 / (classical_energy + 1.0)
            energy_quantum = 1000.0 / (quantum_energy + 1.0)
            # Normalize to 0-100 scale
            energy_max = max(energy_classical, energy_quantum)
            energy_classical_norm = (energy_classical / energy_max) * 100
            energy_quantum_norm = (energy_quantum / energy_max) * 100
            
            # Quality component (30% weight): Quantum advantage probability
            quality_quantum = quantum_advantage * 100
            quality_classical = (1.0 - quantum_advantage) * 100
            
            # Compute weighted scores
            classical_score = (
                0.40 * time_classical_norm +
                0.30 * energy_classical_norm +
                0.30 * quality_classical
            )
            
            quantum_score = (
                0.40 * time_quantum_norm +
                0.30 * energy_quantum_norm +
                0.30 * quality_quantum
            )
            
            score_diff = quantum_score - classical_score
            
            if score_diff > 5.0:
                decision = 'quantum'
                reasoning = (
                    f"BALANCED strategy: Quantum solver scores higher on weighted combination "
                    f"(quantum={quantum_score:.1f}, classical={classical_score:.1f}). "
                    f"Time: {quantum_runtime:.3f}s, Energy: {quantum_energy:.1f}mJ, "
                    f"Quality advantage: {quantum_advantage:.1%}. "
                    f"Optimal balance of performance, efficiency, and solution quality."
                )
            elif score_diff < -5.0:
                decision = 'classical'
                reasoning = (
                    f"BALANCED strategy: Classical solver scores higher on weighted combination "
                    f"(classical={classical_score:.1f}, quantum={quantum_score:.1f}). "
                    f"Time: {classical_runtime:.3f}s, Energy: {classical_energy:.1f}mJ, "
                    f"Quality comparable. "
                    f"Best overall balance of performance, efficiency, and reliability."
                )
            else:
                decision = 'quantum' if quantum_score >= classical_score else 'classical'
                reasoning = (
                    f"BALANCED strategy: Both solvers score similarly "
                    f"(classical={classical_score:.1f}, quantum={quantum_score:.1f}). "
                    f"Selecting {decision} based on slight overall advantage. "
                    f"Trade-offs are well-balanced for this problem."
                )
        
        logger.debug(f"Strategy scoring complete: {decision} (score_diff={score_diff:.2f})")
        
        return (decision, score_diff, reasoning)
    
    def route_problem(
        self,
        problem: ProblemBase,
        edge_env: EdgeEnvironment,
        preferences: Optional[RoutingPreferences] = None
    ) -> Dict[str, Any]:
        """
        Make intelligent routing decision for a problem.
        
        This is the main entry point for routing decisions. It performs
        comprehensive analysis and returns a detailed routing recommendation
        with reasoning, confidence score, and performance estimates.
        
        Decision Algorithm:
        -------------------
        
        **Step 1: Problem Analysis**
        - Extract problem features using ProblemAnalyzer
        - Determine problem size, type, complexity
        - Analyze graph structure (if applicable)
        - Estimate classical and quantum runtimes
        
        **Step 2: Resource Constraint Validation**
        - Create JobRequirements for classical solver
        - Create JobRequirements for quantum solver
        - Check if each solver fits within EdgeEnvironment constraints
        - Identify resource bottlenecks (power, memory, thermal, timeout)
        
        **Step 3: Performance Prediction**
        - Quantum advantage probability (from ProblemAnalyzer)
        - Runtime comparison (classical vs quantum)
        - Energy efficiency comparison (Joules = Watts × Seconds)
        - Solution quality expectations (exact vs approximate)
        
        **Step 4: Decision Tree**
        The router applies a hierarchical decision tree:
        
        ```
        IF problem_size < 10:
            → CLASSICAL (too small for quantum overhead)
        
        ELIF problem_size > 100:
            → CLASSICAL (simulation too slow, real QC needed)
        
        ELIF quantum_resources_unavailable:
            → CLASSICAL (no choice)
        
        ELIF classical_resources_unavailable:
            → QUANTUM (no choice)
        
        ELIF is_bipartite_maxcut:
            → CLASSICAL (polynomial algorithm exists)
        
        ELIF quantum_advantage_high AND quantum_resources_ok:
            IF prefer_speed AND quantum_faster:
                → QUANTUM
            ELIF prefer_energy AND quantum_more_efficient:
                → QUANTUM
            ELIF prefer_quality AND quantum_better_quality:
                → QUANTUM
            ELSE:
                → Consider HYBRID
        
        ELIF power_constrained AND quantum_more_efficient:
            → QUANTUM (better for battery-powered edge devices)
        
        ELIF time_critical AND classical_faster:
            → CLASSICAL (minimize latency)
        
        ELSE:
            → CLASSICAL (safe default)
        ```
        
        **Step 5: Confidence Scoring**
        Confidence (0.0-1.0) reflects certainty of decision:
        - High confidence (>0.8): Strong evidence for decision
        - Medium confidence (0.5-0.8): Reasonable evidence, could go either way
        - Low confidence (<0.5): Weak evidence, significant uncertainty
        
        Factors affecting confidence:
        - Problem size (sweet spot = higher confidence)
        - Resource availability (tight constraints = lower confidence)
        - Quantum advantage probability (higher = more confident in quantum)
        - Historical data (more data = higher confidence, future enhancement)
        
        Args:
            problem: Problem instance to route (must be generated)
            edge_env: EdgeEnvironment with resource constraints
            preferences: Optional user preferences for routing
        
        Returns:
            Dictionary containing:
            {
                'decision': str,              # 'quantum' | 'classical' | 'hybrid'
                'reasoning': str,             # Detailed explanation
                'confidence': float,          # 0.0-1.0 confidence score
                'estimated_time_ms': int,     # Expected runtime in milliseconds
                'estimated_energy_mj': float, # Expected energy in millijoules
                'alternative_options': List[Dict],  # What else could work
                'problem_analysis': Dict,     # Full problem analysis results
                'resource_constraints': Dict, # Resource availability summary
                'performance_predictions': Dict  # Runtime/energy/quality estimates
            }
        
        Raises:
            ValueError: If problem is not generated
            TypeError: If arguments have wrong types
        
        Example:
            >>> from src.problems.maxcut import MaxCutProblem
            >>> from src.router.edge_simulator import EdgeEnvironment, DeploymentProfile
            >>> from src.router.quantum_router import QuantumRouter, RoutingPreferences
            >>> 
            >>> # Create problem
            >>> problem = MaxCutProblem(num_nodes=30)
            >>> problem.generate(edge_probability=0.3)
            >>> 
            >>> # Set up environment
            >>> env = EdgeEnvironment(DeploymentProfile.AEROSPACE)
            >>> 
            >>> # Configure preferences
            >>> prefs = RoutingPreferences(
            ...     prefer_speed=True,
            ...     prefer_energy=True,  # Battery-powered aerospace!
            ...     allow_hybrid=True
            ... )
            >>> 
            >>> # Route problem
            >>> router = QuantumRouter()
            >>> result = router.route_problem(problem, env, prefs)
            >>> 
            >>> print(f"Decision: {result['decision']}")
            >>> print(f"Confidence: {result['confidence']:.1%}")
            >>> print(f"Reasoning: {result['reasoning']}")
            >>> print(f"Estimated time: {result['estimated_time_ms']}ms")
            >>> print(f"Estimated energy: {result['estimated_energy_mj']:.2f}mJ")
        
        Notes:
            - This method is stateless (no side effects)
            - Can be called concurrently from multiple threads
            - Routing decision is deterministic given same inputs
            - Future enhancement: incorporate historical performance data
        """
        
        # Validate inputs
        if not isinstance(problem, ProblemBase):
            raise TypeError(f"problem must be ProblemBase, got {type(problem)}")
        if not problem.is_generated:
            raise ValueError("Problem must be generated before routing")
        if not isinstance(edge_env, EdgeEnvironment):
            raise TypeError(f"edge_env must be EdgeEnvironment, got {type(edge_env)}")
        
        # Use default preferences if not provided
        if preferences is None:
            preferences = RoutingPreferences()
        
        logger.info(f"Routing problem: {problem.problem_type} size={problem.problem_size}")
        
        # =====================================================================
        # STEP 1: ANALYZE PROBLEM
        # =====================================================================
        logger.debug("Step 1: Analyzing problem characteristics")
        
        analysis = self.analyzer.analyze_problem(problem)
        problem_size = analysis['problem_size']
        problem_type = analysis['problem_type']
        
        logger.debug(f"Problem analysis complete: size={problem_size}, type={problem_type}")
        
        # =====================================================================
        # STEP 2: CHECK RESOURCE CONSTRAINTS
        # =====================================================================
        logger.debug("Step 2: Checking resource constraints")
        
        # Estimate resource requirements for classical solver
        classical_job = self._estimate_classical_requirements(analysis, edge_env)
        classical_fits = edge_env.can_execute_classical(classical_job)
        
        # Estimate resource requirements for quantum solver (simulator)
        quantum_job = self._estimate_quantum_requirements(analysis, edge_env)
        quantum_fits = edge_env.can_execute_quantum(quantum_job)
        
        logger.debug(f"Resource check: classical_fits={classical_fits}, quantum_fits={quantum_fits}")
        
        # =====================================================================
        # STEP 3: PREDICT PERFORMANCE
        # =====================================================================
        logger.debug("Step 3: Predicting performance")
        
        quantum_advantage_prob = analysis['quantum_advantage_probability']
        classical_runtime_s = analysis['estimated_classical_runtime']
        quantum_runtime_s = analysis['estimated_quantum_runtime']
        
        # Calculate energy consumption (Energy = Power × Time)
        # Note: This is rough estimate, actual energy depends on many factors
        classical_energy_mj = classical_job.power_watts * classical_runtime_s * 1000  # mJ
        quantum_energy_mj = quantum_job.power_watts * quantum_runtime_s * 1000  # mJ
        
        # Determine which is faster
        quantum_faster = quantum_runtime_s < classical_runtime_s
        quantum_more_efficient = quantum_energy_mj < classical_energy_mj
        
        logger.debug(f"Performance prediction: quantum_advantage_prob={quantum_advantage_prob:.2f}, "
                    f"quantum_faster={quantum_faster}, quantum_efficient={quantum_more_efficient}")
        
        # =====================================================================
        # STEP 4: MAKE DECISION (Decision Tree)
        # =====================================================================
        logger.debug("Step 4: Applying decision tree")
        
        decision = 'classical'  # Default
        reasoning = ""
        confidence = 0.5  # Default medium confidence
        alternative_options = []
        
        # Rule 1: Problem too small
        # Small problems have low quantum overhead relative to benefit
        # Classical solvers are very fast for small instances
        if problem_size < 10:
            decision = 'classical'
            reasoning = (
                f"Problem too small (size={problem_size} < 10). "
                f"Quantum overhead (circuit compilation, measurement statistics) "
                f"outweighs potential speedup. Classical solvers are very fast for "
                f"small problems."
            )
            confidence = 0.95  # High confidence
            
            alternative_options.append({
                'option': 'quantum',
                'feasible': quantum_fits,
                'note': 'Would work but slower due to overhead'
            })
        
        # Rule 2: Problem very large
        # Quantum simulation becomes impractical for >100 qubits
        # Would need real quantum hardware, which has different constraints
        elif problem_size > 100:
            decision = 'classical'
            reasoning = (
                f"Problem very large (size={problem_size} > 100). "
                f"Quantum simulation requires exponential resources (2^{problem_size} state space). "
                f"Classical heuristics are more practical. Real quantum hardware would be needed "
                f"for quantum advantage, but current systems are limited to ~50-1000 qubits with "
                f"high error rates."
            )
            confidence = 0.90
            
            alternative_options.append({
                'option': 'quantum-hardware',
                'feasible': False,
                'note': f'Would need real QC with >{problem_size} qubits (currently limited availability)'
            })
        
        # Rule 3: Quantum resources unavailable
        # If quantum solver can't fit in edge environment, must use classical
        elif not quantum_fits:
            decision = 'classical'
            
            # Determine which resource constraint failed
            if quantum_job.power_watts > edge_env.power_budget_watts:
                constraint = f"power (need {quantum_job.power_watts:.1f}W, have {edge_env.power_budget_watts:.1f}W)"
            elif quantum_job.memory_mb > edge_env.memory_limit_mb:
                constraint = f"memory (need {quantum_job.memory_mb}MB, have {edge_env.memory_limit_mb}MB)"
            elif quantum_job.execution_time_seconds > edge_env.compute_timeout_seconds:
                constraint = f"timeout (need {quantum_job.execution_time_seconds:.1f}s, have {edge_env.compute_timeout_seconds:.1f}s)"
            else:
                constraint = "resource constraints"
            
            reasoning = (
                f"Quantum solver exceeds {constraint} for {edge_env.profile.value} deployment. "
                f"Rotonium's room-temp QPU eliminates cryogenic cooling, but edge environments "
                f"still have power, memory, and thermal constraints. Classical solver is only option."
            )
            confidence = 1.0  # Certain (no choice)
            
            alternative_options.append({
                'option': 'quantum',
                'feasible': False,
                'note': f'Blocked by {constraint}'
            })
        
        # Rule 4: Classical resources unavailable
        # If classical solver can't fit, must try quantum (rare case)
        elif not classical_fits:
            decision = 'quantum'
            reasoning = (
                f"Classical solver exceeds resource constraints for {edge_env.profile.value} deployment. "
                f"Quantum solver is only available option."
            )
            confidence = 1.0  # Certain (no choice)
            
            alternative_options.append({
                'option': 'classical',
                'feasible': False,
                'note': 'Blocked by resource constraints'
            })
        
        # Rule 5: Special case - Bipartite MaxCut
        # Bipartite MaxCut is in P (polynomial time), not NP-hard!
        # Classical solver can find optimal solution efficiently
        elif problem_type == 'maxcut':
            try:
                graph = problem.to_graph()
                import networkx as nx
                if nx.is_bipartite(graph):
                    decision = 'classical'
                    reasoning = (
                        f"Problem is bipartite MaxCut, which is in P (polynomial time). "
                        f"Classical algorithm can find optimal solution efficiently. "
                        f"No quantum advantage for this special case."
                    )
                    confidence = 0.99  # Very high
                    
                    alternative_options.append({
                        'option': 'quantum',
                        'feasible': quantum_fits,
                        'note': 'Would work but no advantage for bipartite case'
                    })
            except:
                pass  # Not a graph problem or check failed
        
        # Rule 6: High quantum advantage predicted
        # Quantum solver likely to outperform classical
        if decision == 'classical' and quantum_advantage_prob > 0.6 and quantum_fits:
            # Consider user preferences
            should_use_quantum = False
            
            if preferences.prefer_speed and quantum_faster:
                should_use_quantum = True
                reason_detail = f"quantum is faster ({quantum_runtime_s:.3f}s vs {classical_runtime_s:.3f}s)"
            elif preferences.prefer_energy and quantum_more_efficient:
                should_use_quantum = True
                reason_detail = f"quantum is more efficient ({quantum_energy_mj:.2f}mJ vs {classical_energy_mj:.2f}mJ)"
            elif quantum_advantage_prob > 0.7:
                should_use_quantum = True
                reason_detail = f"high quantum advantage probability ({quantum_advantage_prob:.1%})"
            
            if should_use_quantum:
                decision = 'quantum'
                reasoning = (
                    f"Quantum solver predicted to have advantage ({reason_detail}). "
                    f"Problem size ({problem_size}) in quantum sweet spot (10-100 variables). "
                    f"Rotonium's room-temp QPU enables efficient quantum computation on edge device."
                )
                confidence = 0.6 + (quantum_advantage_prob - 0.6) * 0.5  # 0.6-0.8 range
                
                alternative_options.append({
                    'option': 'classical',
                    'feasible': classical_fits,
                    'note': f'Would work but likely slower or lower quality'
                })
                
                # Also suggest hybrid if allowed
                if preferences.allow_hybrid:
                    alternative_options.append({
                        'option': 'hybrid',
                        'feasible': True,
                        'note': 'Use quantum for initial solution, classical for refinement'
                    })
        
        # Rule 7: Power constrained and quantum more efficient
        # Important for battery-powered edge devices (aerospace, mobile)
        if decision == 'classical' and quantum_fits and quantum_more_efficient:
            if preferences.prefer_energy:
                decision = 'quantum'
                reasoning = (
                    f"Power-constrained {edge_env.profile.value} deployment prioritizes energy efficiency. "
                    f"Quantum solver uses less energy ({quantum_energy_mj:.2f}mJ vs {classical_energy_mj:.2f}mJ). "
                    f"Critical for battery-powered edge devices. Rotonium's low-power QPU (<50W) "
                    f"enables longer battery life compared to classical computation."
                )
                confidence = 0.70
                
                alternative_options.append({
                    'option': 'classical',
                    'feasible': classical_fits,
                    'note': f'Uses more energy ({classical_energy_mj:.2f}mJ), drains battery faster'
                })
        
        # Rule 8: Time critical and classical faster
        # Real-time applications (navigation, sensing) need fast response
        if decision == 'quantum' and preferences.prefer_speed and not quantum_faster:
            decision = 'classical'
            reasoning = (
                f"Time-critical application requires fast response. "
                f"Classical solver is faster ({classical_runtime_s:.3f}s vs {quantum_runtime_s:.3f}s). "
                f"Quantum simulation overhead outweighs potential benefits for this problem size."
            )
            confidence = 0.75
            
            alternative_options.append({
                'option': 'quantum',
                'feasible': quantum_fits,
                'note': f'Would work but slower ({quantum_runtime_s:.3f}s)'
            })
        
        # Rule 9: Strategy-based scoring
        # Apply strategy-specific scoring if decision not yet made with high confidence
        # This allows strategy to override default logic when appropriate
        if decision != 'quantum' and decision != 'hybrid' and confidence < 0.8:
            if quantum_fits and classical_fits:
                # Both solvers fit - use strategy to make final decision
                logger.debug("Applying strategy-based scoring")
                
                # Prepare estimates for scoring
                classical_estimates = {
                    'runtime_s': classical_runtime_s,
                    'energy_mj': classical_energy_mj,
                    'fits_resources': classical_fits
                }
                
                quantum_estimates = {
                    'runtime_s': quantum_runtime_s,
                    'energy_mj': quantum_energy_mj,
                    'fits_resources': quantum_fits,
                    'quantum_advantage_prob': quantum_advantage_prob
                }
                
                # Calculate strategy-based scores
                strategy_decision, score_diff, strategy_reasoning = self.calculate_strategy_score(
                    classical_estimates,
                    quantum_estimates,
                    self.current_strategy
                )
                
                # If strategy has strong preference (score_diff > 5), override default
                if abs(score_diff) > 5.0:
                    decision = strategy_decision
                    reasoning = strategy_reasoning
                    confidence = min(0.85, 0.65 + abs(score_diff) / 100)  # Higher score diff = higher confidence
                    
                    logger.info(f"Strategy override: {decision} selected by {self.current_strategy.value} "
                               f"strategy (score_diff={score_diff:.2f})")
                    
                    # Add alternative as "what strategy didn't choose"
                    alt_decision = 'classical' if decision == 'quantum' else 'quantum'
                    alternative_options.append({
                        'option': alt_decision,
                        'feasible': True,
                        'note': f'Not selected by {self.current_strategy.value} strategy'
                    })
        
        # Rule 10: Default to classical if no strong evidence for quantum
        # Classical solvers are mature, predictable, and well-understood
        if decision != 'quantum' and decision != 'hybrid':
            if not reasoning:  # No specific reason assigned yet
                decision = 'classical'
                reasoning = (
                    f"Classical solver selected as safe default. "
                    f"Problem size={problem_size}, quantum advantage probability={quantum_advantage_prob:.1%}. "
                    f"Classical solvers are mature and predictable. Quantum advantage not strong enough "
                    f"to justify quantum routing."
                )
                confidence = 0.55
                
                if quantum_fits:
                    alternative_options.append({
                        'option': 'quantum',
                        'feasible': True,
                        'note': 'Could work but confidence is low'
                    })
        
        # Suggest hybrid if both solvers fit and confidence is medium
        if preferences.allow_hybrid and quantum_fits and classical_fits:
            if 0.4 < confidence < 0.7:
                alternative_options.append({
                    'option': 'hybrid',
                    'feasible': True,
                    'note': 'Run both solvers in parallel or use quantum initialization with classical refinement'
                })
        
        # =====================================================================
        # STEP 5: COMPILE RESULTS
        # =====================================================================
        logger.debug(f"Decision: {decision}, confidence={confidence:.2f}")
        
        # Convert runtimes to milliseconds
        estimated_time_ms = (quantum_runtime_s if decision == 'quantum' else classical_runtime_s) * 1000
        estimated_energy_mj = quantum_energy_mj if decision == 'quantum' else classical_energy_mj
        
        result = {
            'decision': decision,
            'reasoning': reasoning,
            'confidence': confidence,
            'estimated_time_ms': int(estimated_time_ms),
            'estimated_energy_mj': estimated_energy_mj,
            'alternative_options': alternative_options,
            'strategy_used': self.current_strategy.value,  # Add strategy info
            
            # Additional detailed information
            'problem_analysis': {
                'problem_type': problem_type,
                'problem_size': problem_size,
                'complexity': analysis['complexity_estimate'],
                'quantum_advantage_probability': quantum_advantage_prob,
                'suitability_scores': analysis['suitability_scores']
            },
            
            'resource_constraints': {
                'environment_profile': edge_env.profile.value,
                'classical_fits': classical_fits,
                'quantum_fits': quantum_fits,
                'power_budget_watts': edge_env.power_budget_watts,
                'memory_limit_mb': edge_env.memory_limit_mb,
                'timeout_seconds': edge_env.compute_timeout_seconds
            },
            
            'performance_predictions': {
                'classical_runtime_seconds': classical_runtime_s,
                'quantum_runtime_seconds': quantum_runtime_s,
                'classical_energy_mj': classical_energy_mj,
                'quantum_energy_mj': quantum_energy_mj,
                'quantum_faster': quantum_faster,
                'quantum_more_efficient': quantum_more_efficient
            }
        }
        
        logger.info(f"Routing complete: {decision} (confidence={confidence:.1%})")
        
        return result
    
    def explain_decision(self, routing_result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of routing decision.
        
        This method takes a routing result and formats it into a clear,
        comprehensive explanation suitable for:
        - User interfaces (showing why decision was made)
        - Logging and debugging (understanding routing behavior)
        - Reports and documentation (transparent decision making)
        
        The explanation includes:
        - Primary decision and confidence
        - Detailed reasoning
        - Problem characteristics
        - Resource constraints
        - Performance predictions
        - Alternative options
        
        Args:
            routing_result: Result dictionary from route_problem()
        
        Returns:
            Formatted explanation as multi-line string
        
        Example:
            >>> result = router.route_problem(problem, env, prefs)
            >>> explanation = router.explain_decision(result)
            >>> print(explanation)
            
            ================================================================================
            ROUTING DECISION EXPLANATION
            ================================================================================
            
            DECISION: QUANTUM
            CONFIDENCE: 75.0%
            
            REASONING:
            Quantum solver predicted to have advantage (quantum is faster (2.145s vs 5.234s)).
            Problem size (30) in quantum sweet spot (10-100 variables).
            Rotonium's room-temp QPU enables efficient quantum computation on edge device.
            ...
        
        Notes:
            - Explanation is formatted for 80-column terminal display
            - Includes all relevant information for understanding decision
            - Safe to call multiple times (stateless)
        """
        
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("ROUTING DECISION EXPLANATION")
        lines.append("=" * 80)
        lines.append("")
        
        # Main decision
        decision = routing_result['decision'].upper()
        confidence = routing_result['confidence'] * 100
        strategy = routing_result.get('strategy_used', 'balanced')
        
        lines.append(f"DECISION: {decision}")
        lines.append(f"STRATEGY: {strategy.upper()}")
        lines.append(f"CONFIDENCE: {confidence:.1f}%")
        lines.append("")
        
        # Confidence interpretation
        if confidence >= 80:
            lines.append("Confidence Level: HIGH - Strong evidence supports this decision")
        elif confidence >= 50:
            lines.append("Confidence Level: MEDIUM - Reasonable evidence, some uncertainty")
        else:
            lines.append("Confidence Level: LOW - Significant uncertainty, consider alternatives")
        lines.append("")
        
        # Reasoning
        lines.append("REASONING:")
        lines.append("-" * 80)
        # Wrap long reasoning text
        reasoning = routing_result['reasoning']
        words = reasoning.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= 78:
                current_line += (" " if current_line else "") + word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        lines.append("")
        
        # Problem Analysis
        lines.append("PROBLEM ANALYSIS:")
        lines.append("-" * 80)
        pa = routing_result['problem_analysis']
        lines.append(f"Type: {pa['problem_type']}")
        lines.append(f"Size: {pa['problem_size']} variables")
        lines.append(f"Complexity: {pa['complexity']}")
        lines.append(f"Quantum Advantage Probability: {pa['quantum_advantage_probability']:.1%}")
        lines.append(f"Classical Suitability: {pa['suitability_scores']['classical_score']:.2f}")
        lines.append(f"Quantum Suitability: {pa['suitability_scores']['quantum_score']:.2f}")
        lines.append("")
        
        # Resource Constraints
        lines.append("RESOURCE CONSTRAINTS:")
        lines.append("-" * 80)
        rc = routing_result['resource_constraints']
        lines.append(f"Environment: {rc['environment_profile']}")
        lines.append(f"Power Budget: {rc['power_budget_watts']}W")
        lines.append(f"Memory Limit: {rc['memory_limit_mb']}MB")
        lines.append(f"Timeout: {rc['timeout_seconds']}s")
        lines.append(f"Classical Solver Fits: {'Yes' if rc['classical_fits'] else 'No'}")
        lines.append(f"Quantum Solver Fits: {'Yes' if rc['quantum_fits'] else 'No'}")
        lines.append("")
        
        # Performance Predictions
        lines.append("PERFORMANCE PREDICTIONS:")
        lines.append("-" * 80)
        pp = routing_result['performance_predictions']
        lines.append(f"Classical Runtime: {pp['classical_runtime_seconds']:.3f}s")
        lines.append(f"Quantum Runtime: {pp['quantum_runtime_seconds']:.3f}s")
        lines.append(f"Classical Energy: {pp['classical_energy_mj']:.2f}mJ")
        lines.append(f"Quantum Energy: {pp['quantum_energy_mj']:.2f}mJ")
        
        if pp['quantum_faster']:
            speedup = pp['classical_runtime_seconds'] / pp['quantum_runtime_seconds']
            lines.append(f"Quantum is FASTER by {speedup:.2f}x")
        else:
            slowdown = pp['quantum_runtime_seconds'] / pp['classical_runtime_seconds']
            lines.append(f"Classical is FASTER by {slowdown:.2f}x")
        
        if pp['quantum_more_efficient']:
            efficiency = pp['classical_energy_mj'] / pp['quantum_energy_mj']
            lines.append(f"Quantum is MORE EFFICIENT by {efficiency:.2f}x")
        else:
            inefficiency = pp['quantum_energy_mj'] / pp['classical_energy_mj']
            lines.append(f"Classical is MORE EFFICIENT by {inefficiency:.2f}x")
        lines.append("")
        
        # Alternative Options
        if routing_result['alternative_options']:
            lines.append("ALTERNATIVE OPTIONS:")
            lines.append("-" * 80)
            for i, alt in enumerate(routing_result['alternative_options'], 1):
                feasible = "Feasible" if alt['feasible'] else "Not Feasible"
                lines.append(f"{i}. {alt['option'].upper()} ({feasible})")
                lines.append(f"   Note: {alt['note']}")
            lines.append("")
        
        # Selected option summary
        lines.append("SELECTED OPTION:")
        lines.append("-" * 80)
        lines.append(f"Solver: {decision}")
        lines.append(f"Estimated Time: {routing_result['estimated_time_ms']}ms")
        lines.append(f"Estimated Energy: {routing_result['estimated_energy_mj']:.2f}mJ")
        lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF EXPLANATION")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def suggest_alternatives(self, routing_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest alternative routing options with parameter adjustments.
        
        This method analyzes a routing result and suggests "what if" scenarios:
        - What if we had more power budget?
        - What if we had more time?
        - What if we used different preferences?
        
        Useful for:
        - Capacity planning (what hardware upgrades would help?)
        - Trade-off analysis (speed vs energy vs quality)
        - User education (showing impact of different choices)
        
        Suggestions include:
        - Required resource adjustments
        - Expected performance improvements
        - Feasibility assessment
        - Cost-benefit analysis
        
        Args:
            routing_result: Result dictionary from route_problem()
        
        Returns:
            List of suggestion dictionaries, each containing:
            {
                'suggestion_type': str,         # Type of adjustment
                'description': str,             # Human-readable description
                'required_changes': Dict,       # What needs to change
                'expected_improvements': Dict,  # What would improve
                'feasibility': str,            # 'easy', 'moderate', 'difficult'
                'recommendation': str          # Should you do this?
            }
        
        Example:
            >>> result = router.route_problem(problem, env, prefs)
            >>> suggestions = router.suggest_alternatives(result)
            >>> 
            >>> for suggestion in suggestions:
            ...     print(f"Suggestion: {suggestion['suggestion_type']}")
            ...     print(f"Description: {suggestion['description']}")
            ...     print(f"Feasibility: {suggestion['feasibility']}")
            ...     print(f"Recommendation: {suggestion['recommendation']}")
            ...     print()
        
        Notes:
            - Suggestions are ordered by feasibility (easiest first)
            - All suggestions are hypothetical (don't change actual system)
            - Use for planning and analysis, not automatic execution
        """
        
        suggestions = []
        
        decision = routing_result['decision']
        pp = routing_result['performance_predictions']
        rc = routing_result['resource_constraints']
        
        # Suggestion 1: Increase power budget
        # Useful if quantum solver was rejected due to power constraints
        if decision == 'classical' and not rc['quantum_fits']:
            # Check if power was the limiting factor
            # (This is approximate - would need actual resource check details)
            suggestions.append({
                'suggestion_type': 'increase_power_budget',
                'description': 'Increase power budget to enable quantum solver',
                'required_changes': {
                    'power_budget_watts': rc['power_budget_watts'] * 1.5,
                    'reasoning': 'Quantum solver requires more power due to additional qubits and gates'
                },
                'expected_improvements': {
                    'quantum_feasible': True,
                    'potential_speedup': pp['classical_runtime_seconds'] / pp['quantum_runtime_seconds']
                                        if pp['quantum_runtime_seconds'] > 0 else 1.0
                },
                'feasibility': 'moderate',
                'recommendation': (
                    'Consider if speed or solution quality improvements justify increased power consumption. '
                    'For battery-powered devices, evaluate impact on battery life. '
                    'Rotonium QPU already optimized for low power, but larger problems need more resources.'
                )
            })
        
        # Suggestion 2: Increase timeout
        # Allows more thorough search, better solution quality
        if rc['timeout_seconds'] < 60:
            speedup = 1.5 if decision == 'quantum' else 1.3
            suggestions.append({
                'suggestion_type': 'increase_timeout',
                'description': f'Increase computation timeout from {rc["timeout_seconds"]}s to 60s',
                'required_changes': {
                    'timeout_seconds': 60.0,
                    'reasoning': 'More time allows deeper search, iterative refinement, better solution quality'
                },
                'expected_improvements': {
                    'solution_quality': '+10-20% improvement',
                    'algorithm_options': 'Can use more sophisticated algorithms (QAOA with more layers, classical branch-and-bound)'
                },
                'feasibility': 'easy',
                'recommendation': (
                    'Recommended if not time-critical. Batch processing applications can benefit greatly. '
                    'For real-time applications (navigation, sensing), keep tight timeout.'
                )
            })
        
        # Suggestion 3: Hybrid approach
        # Use quantum for initialization, classical for refinement
        if routing_result['alternative_options']:
            for alt in routing_result['alternative_options']:
                if alt['option'] == 'hybrid' and alt['feasible']:
                    suggestions.append({
                        'suggestion_type': 'hybrid_approach',
                        'description': 'Use quantum-classical hybrid solver',
                        'required_changes': {
                            'solver_type': 'hybrid',
                            'approach': 'Quantum for initial solution, classical for local refinement'
                        },
                        'expected_improvements': {
                            'solution_quality': 'Best of both worlds - quantum exploration + classical exploitation',
                            'robustness': 'Fallback to classical if quantum fails',
                            'runtime': f'Estimated {routing_result["estimated_time_ms"] * 1.2:.0f}ms (slightly longer but better quality)'
                        },
                        'feasibility': 'easy',
                        'recommendation': (
                            'Highly recommended for production systems. Provides robustness and often best solution quality. '
                            'Slight time overhead (~20%) but significant quality improvement (~30-50%).'
                        )
                    })
        
        # Suggestion 4: Use different deployment profile
        # Move to more capable edge server if available
        if rc['environment_profile'] in ['aerospace', 'mobile']:
            suggestions.append({
                'suggestion_type': 'upgrade_deployment',
                'description': f'Upgrade from {rc["environment_profile"]} to ground_server',
                'required_changes': {
                    'environment_profile': 'ground_server',
                    'power_budget': '200W (vs current {})W'.format(rc['power_budget_watts']),
                    'memory': '16GB (vs current {})MB'.format(rc['memory_limit_mb'])
                },
                'expected_improvements': {
                    'problem_size_limit': '2-3x larger problems can be solved',
                    'solution_quality': 'Can afford more expensive algorithms',
                    'reliability': 'More stable power, better cooling'
                },
                'feasibility': 'difficult',
                'recommendation': (
                    'Consider if problem regularly exceeds current resources. '
                    'Trade-off: lose edge deployment benefits (low latency, local processing). '
                    'May be worth it for complex optimization problems.'
                )
            })
        
        # Suggestion 5: Reduce problem size
        # Decompose or sample problem to fit constraints
        if routing_result['problem_analysis']['problem_size'] > 50:
            suggestions.append({
                'suggestion_type': 'problem_decomposition',
                'description': 'Decompose problem into smaller subproblems',
                'required_changes': {
                    'problem_size': f"Reduce from {routing_result['problem_analysis']['problem_size']} to 30-40 variables per subproblem",
                    'approach': 'Divide-and-conquer or hierarchical decomposition'
                },
                'expected_improvements': {
                    'quantum_feasible': 'Subproblems fit in quantum sweet spot',
                    'parallelization': 'Can solve subproblems in parallel',
                    'scalability': 'Can handle arbitrarily large problems'
                },
                'feasibility': 'moderate',
                'recommendation': (
                    'Recommended for large problems (>50 variables). '
                    'Works well if problem has natural decomposition structure (e.g., multiple vehicles, regions). '
                    'May sacrifice some global optimality for practical solvability.'
                )
            })
        
        # Suggestion 6: Prefer energy efficiency
        # Important for battery-powered edge devices
        if decision == 'classical' and pp['quantum_more_efficient']:
            suggestions.append({
                'suggestion_type': 'optimize_for_energy',
                'description': 'Switch to quantum solver to reduce energy consumption',
                'required_changes': {
                    'preference': 'Set prefer_energy=True in routing preferences',
                    'decision': 'Would route to quantum'
                },
                'expected_improvements': {
                    'energy_savings': f"{(pp['classical_energy_mj'] - pp['quantum_energy_mj']):.2f}mJ "
                                     f"({(1 - pp['quantum_energy_mj']/pp['classical_energy_mj'])*100:.1f}% reduction)",
                    'battery_life': 'Extends battery life for mobile/aerospace deployments',
                    'thermal_load': 'Reduces heat generation'
                },
                'feasibility': 'easy',
                'recommendation': (
                    'Strongly recommended for battery-powered edge devices (aerospace, mobile). '
                    'Rotonium QPU power efficiency (<50W) vs classical processing can significantly extend '
                    'mission duration. Small time trade-off for major energy savings.'
                )
            })
        
        # Sort suggestions by feasibility (easiest first)
        feasibility_order = {'easy': 0, 'moderate': 1, 'difficult': 2}
        suggestions.sort(key=lambda x: feasibility_order.get(x['feasibility'], 999))
        
        return suggestions
    
    # =========================================================================
    # Historical Learning Methods
    # =========================================================================
    
    def update_performance_model(self, job_execution: JobExecution) -> None:
        """
        Update performance model after job completion.
        
        This method learns from actual execution results to improve future
        routing decisions. By comparing predictions with reality, the router
        becomes more accurate over time.
        
        Learning Process:
        -----------------
        1. **Record Execution**: Add to execution_history
        2. **Calculate Errors**: Compare predicted vs actual metrics
        3. **Invalidate Cache**: Clear cached statistics for affected category
        4. **Adjust Thresholds**: Trigger threshold adjustment if enough data
        
        Prediction Errors Tracked:
        --------------------------
        - **Time Error**: |actual_time - estimated_time|
        - **Energy Error**: |actual_energy - estimated_energy|
        - **Quality Gap**: Difference between expected and achieved quality
        
        These errors help calibrate:
        - Runtime estimation models
        - Energy consumption models
        - Quantum advantage predictions
        - Confidence scoring
        
        Use Cases:
        ----------
        - **After classical solver completes**: Update classical performance model
        - **After quantum solver completes**: Update quantum performance model
        - **After hybrid execution**: Update both models with their portions
        - **On failure**: Record failure patterns to avoid repeating
        
        Learning Improvements:
        ----------------------
        With sufficient data (typically 10+ executions per category):
        - Routing becomes more accurate for specific problem types/sizes
        - Quantum advantage predictions improve
        - Resource estimates become more reliable
        - Confidence scores better reflect reality
        
        Args:
            job_execution: Completed job execution record with actuals
        
        Example:
            >>> # After job completes
            >>> execution = JobExecution(
            ...     job_id="job_123",
            ...     timestamp=datetime.now(),
            ...     problem_type="maxcut",
            ...     problem_size=30,
            ...     solver_used="quantum",
            ...     estimated_time_s=3.0,
            ...     estimated_energy_mj=120.0,
            ...     predicted_quantum_advantage=0.65,
            ...     actual_time_s=2.8,      # Actual measurement
            ...     actual_energy_mj=115.0, # Actual measurement
            ...     solution_quality=0.92,  # Quality achieved
            ...     success=True
            ... )
            >>> router.update_performance_model(execution)
            >>> # Future routing for similar problems will be more accurate
        
        Notes:
            - Only updates if learning_enabled=True
            - Requires actual measurements (actual_time_s, actual_energy_mj)
            - Cache invalidation ensures statistics recalculated on next query
            - Threshold adjustment happens periodically (not every update)
        """
        
        if not self.learning_enabled:
            logger.debug("Learning disabled, skipping performance model update")
            return
        
        # Validate execution has actual measurements
        if job_execution.actual_time_s is None or job_execution.actual_energy_mj is None:
            logger.warning(f"Job {job_execution.job_id} missing actual measurements, "
                          "cannot update performance model")
            return
        
        # Add to history
        self.execution_history.append(job_execution)
        logger.info(f"Recorded execution: {job_execution.job_id} ({job_execution.solver_used}, "
                   f"{job_execution.problem_type}, size={job_execution.problem_size})")
        
        # Calculate prediction errors
        time_error = abs(job_execution.actual_time_s - job_execution.estimated_time_s)
        energy_error = abs(job_execution.actual_energy_mj - job_execution.estimated_energy_mj)
        time_error_pct = (time_error / job_execution.estimated_time_s) * 100
        energy_error_pct = (energy_error / job_execution.estimated_energy_mj) * 100
        
        logger.debug(f"Prediction errors: time={time_error_pct:.1f}%, energy={energy_error_pct:.1f}%")
        
        # Invalidate cached statistics for this category
        # This ensures fresh statistics are computed on next query
        size_range = self._get_size_range(job_execution.problem_size)
        cache_key = (job_execution.problem_type, size_range, job_execution.solver_used)
        
        if cache_key in self.performance_cache:
            del self.performance_cache[cache_key]
            logger.debug(f"Invalidated cache for {cache_key}")
        
        # Periodically adjust routing thresholds based on accumulated history
        # Do this every 10 executions to avoid overhead
        if len(self.execution_history) % 10 == 0:
            self.adjust_routing_thresholds()
            logger.info(f"Adjusted routing thresholds after {len(self.execution_history)} executions")
        
        # Log summary statistics
        if len(self.execution_history) >= 10:
            recent = self.execution_history[-10:]
            avg_time_error = statistics.mean([abs(e.actual_time_s - e.estimated_time_s) 
                                              for e in recent if e.actual_time_s])
            success_rate = sum(1 for e in recent if e.success) / len(recent)
            
            logger.info(f"Recent performance: avg_time_error={avg_time_error:.3f}s, "
                       f"success_rate={success_rate:.1%}")
    
    def get_historical_performance(
        self,
        problem_type: str,
        size_range: Tuple[int, int]
    ) -> Dict[str, PerformanceStatistics]:
        """
        Query historical performance for similar past jobs.
        
        This method retrieves aggregated statistics from previous executions
        matching the specified problem type and size range. These statistics
        inform routing decisions for new problems.
        
        Statistics Computed:
        --------------------
        For both classical and quantum solvers (when data available):
        
        - **Execution Metrics**:
          * avg_time_s: Mean execution time
          * std_time_s: Standard deviation (variability)
          * avg_energy_mj: Mean energy consumption
          * std_energy_mj: Energy variability
          * avg_quality: Mean solution quality achieved
        
        - **Reliability Metrics**:
          * success_rate: Fraction of jobs that completed successfully
          * num_executions: Sample size (higher = more reliable stats)
        
        - **Prediction Accuracy**:
          * avg_prediction_error_time: Mean |actual - predicted| time
          * avg_prediction_error_energy: Mean |actual - predicted| energy
        
        - **Comparative Metrics**:
          * avg_quantum_advantage: Observed speedup (quantum time / classical time)
        
        Size Range Categorization:
        --------------------------
        Problems are grouped into size ranges for statistical aggregation:
        - Small: 1-20 variables
        - Medium: 21-50 variables
        - Large: 51-100 variables
        - Very Large: 101+ variables
        
        This grouping ensures sufficient sample size while maintaining relevance.
        
        Caching:
        --------
        Results are cached until new executions are recorded in the same
        category. This avoids expensive recomputation on every query.
        
        Args:
            problem_type: Problem type to query ('maxcut', 'tsp', 'portfolio', etc.)
            size_range: (min, max) problem size range to query
        
        Returns:
            Dictionary with statistics for each solver:
            {
                'classical': PerformanceStatistics or None,
                'quantum': PerformanceStatistics or None,
                'comparison': {
                    'quantum_faster': bool,
                    'quantum_more_efficient': bool,
                    'avg_speedup': float,
                    'avg_energy_ratio': float
                }
            }
        
        Example:
            >>> # Query historical performance for medium MaxCut problems
            >>> stats = router.get_historical_performance('maxcut', (21, 50))
            >>> 
            >>> if stats['quantum']:
            ...     print(f"Quantum average time: {stats['quantum'].avg_time_s:.3f}s")
            ...     print(f"Success rate: {stats['quantum'].success_rate:.1%}")
            ...     print(f"Sample size: {stats['quantum'].num_executions}")
            >>> 
            >>> if stats['comparison']['quantum_faster']:
            ...     speedup = stats['comparison']['avg_speedup']
            ...     print(f"Quantum is {speedup:.2f}x faster on average")
        
        Notes:
            - Returns None for solvers with no historical data
            - Requires at least 3 executions for reliable statistics
            - Statistics improve with more data (10+ executions ideal)
            - Cached until new data added to this category
        """
        
        # Check cache first
        classical_key = (problem_type, size_range, 'classical')
        quantum_key = (problem_type, size_range, 'quantum')
        
        classical_stats = self.performance_cache.get(classical_key)
        quantum_stats = self.performance_cache.get(quantum_key)
        
        # If not cached, compute from history
        if classical_stats is None:
            classical_stats = self._compute_statistics(problem_type, size_range, 'classical')
            if classical_stats and classical_stats.num_executions >= 3:
                self.performance_cache[classical_key] = classical_stats
        
        if quantum_stats is None:
            quantum_stats = self._compute_statistics(problem_type, size_range, 'quantum')
            if quantum_stats and quantum_stats.num_executions >= 3:
                self.performance_cache[quantum_key] = quantum_stats
        
        # Compute comparison metrics if both have data
        comparison = {}
        if classical_stats and quantum_stats:
            if classical_stats.num_executions >= 3 and quantum_stats.num_executions >= 3:
                quantum_faster = quantum_stats.avg_time_s < classical_stats.avg_time_s
                quantum_more_efficient = quantum_stats.avg_energy_mj < classical_stats.avg_energy_mj
                
                avg_speedup = classical_stats.avg_time_s / quantum_stats.avg_time_s if quantum_stats.avg_time_s > 0 else 1.0
                avg_energy_ratio = classical_stats.avg_energy_mj / quantum_stats.avg_energy_mj if quantum_stats.avg_energy_mj > 0 else 1.0
                
                comparison = {
                    'quantum_faster': quantum_faster,
                    'quantum_more_efficient': quantum_more_efficient,
                    'avg_speedup': avg_speedup,
                    'avg_energy_ratio': avg_energy_ratio
                }
        
        result = {
            'classical': classical_stats,
            'quantum': quantum_stats,
            'comparison': comparison
        }
        
        logger.debug(f"Historical performance query: {problem_type}, size_range={size_range}, "
                    f"classical_execs={classical_stats.num_executions if classical_stats else 0}, "
                    f"quantum_execs={quantum_stats.num_executions if quantum_stats else 0}")
        
        return result
    
    def adjust_routing_thresholds(self) -> None:
        """
        Dynamically adjust routing thresholds based on historical performance.
        
        This method analyzes accumulated execution history to fine-tune
        routing decision thresholds. As the router gains experience, it
        learns which problem sizes and types benefit from quantum vs classical.
        
        Thresholds Adjusted:
        --------------------
        
        1. **min_problem_size**:
           - Default: 10 (below this, use classical)
           - Adjustment: Lower if quantum shows advantage for small problems
           - Example: If quantum consistently faster for size 8-10, lower to 8
        
        2. **max_problem_size**:
           - Default: 100 (above this, use classical due to simulation limits)
           - Adjustment: Raise if quantum simulation performs well
           - Example: If size 100-120 quantum executions successful, raise to 120
        
        3. **quantum_advantage_threshold**:
           - Default: 0.6 (require 60% confidence for quantum routing)
           - Adjustment: Lower if quantum consistently outperforms predictions
           - Example: If quantum advantage realized 80% of time, lower threshold
        
        4. **confidence_threshold**:
           - Default: 0.5 (minimum confidence to trust decision)
           - Adjustment: Raise if low-confidence decisions often wrong
           - Example: If success rate < 70% for confidence < 0.6, raise threshold
        
        Learning Logic:
        ---------------
        
        For each problem type and size range:
        - Compare actual quantum advantage vs predicted
        - If quantum consistently better than predicted → lower quantum threshold
        - If classical consistently better than predicted → raise quantum threshold
        - Adjust size boundaries based on success patterns
        
        Stability:
        ----------
        - Requires minimum 20 total executions before adjusting
        - Changes are gradual (max 10% per adjustment)
        - Prevents over-fitting to recent executions
        - Logs all threshold changes for auditability
        
        Example Adjustments:
        --------------------
        
        **Scenario 1**: Quantum outperforming predictions
        ```
        Before: quantum_advantage_threshold = 0.6
        Observation: 15/20 quantum jobs faster than predicted
        After: quantum_advantage_threshold = 0.55 (lowered by 5%)
        Effect: Router more willing to try quantum
        ```
        
        **Scenario 2**: Small problems quantum-friendly
        ```
        Before: min_problem_size = 10
        Observation: Size 8-10 quantum success rate 90%
        After: min_problem_size = 8 (lowered by 2)
        Effect: Router considers quantum for smaller problems
        ```
        
        **Scenario 3**: Classical better than expected
        ```
        Before: quantum_advantage_threshold = 0.6
        Observation: 12/20 classical jobs better than quantum
        After: quantum_advantage_threshold = 0.65 (raised by 5%)
        Effect: Router more conservative about quantum
        ```
        
        Notes:
            - Only adjusts if learning_enabled=True
            - Requires minimum sample size (20 executions)
            - Changes are logged for transparency
            - Thresholds can be manually reset if needed
        """
        
        if not self.learning_enabled:
            return
        
        # Need minimum history to adjust thresholds
        if len(self.execution_history) < 20:
            logger.debug(f"Insufficient history for threshold adjustment "
                        f"({len(self.execution_history)} < 20)")
            return
        
        logger.info("Adjusting routing thresholds based on historical performance")
        
        # Group executions by problem type
        quantum_executions = [e for e in self.execution_history if e.solver_used == 'quantum']
        classical_executions = [e for e in self.execution_history if e.solver_used == 'classical']
        
        if not quantum_executions:
            logger.debug("No quantum executions in history, skipping threshold adjustment")
            return
        
        # =====================================================================
        # Adjust quantum_advantage_threshold
        # =====================================================================
        # If quantum consistently outperforms classical, lower threshold
        # If classical consistently better, raise threshold
        
        successful_quantum = [e for e in quantum_executions if e.success]
        if successful_quantum:
            # Compare quantum actual time vs what classical would have been
            # (approximate using problem size scaling)
            quantum_better_count = 0
            quantum_worse_count = 0
            
            for qe in successful_quantum:
                # Find comparable classical executions (same type, similar size)
                comparable_classical = [
                    ce for ce in classical_executions
                    if ce.problem_type == qe.problem_type
                    and abs(ce.problem_size - qe.problem_size) <= 5
                    and ce.success
                ]
                
                if comparable_classical:
                    avg_classical_time = statistics.mean([ce.actual_time_s for ce in comparable_classical])
                    if qe.actual_time_s < avg_classical_time:
                        quantum_better_count += 1
                    else:
                        quantum_worse_count += 1
            
            total_comparisons = quantum_better_count + quantum_worse_count
            if total_comparisons >= 10:
                quantum_win_rate = quantum_better_count / total_comparisons
                
                old_threshold = self.routing_thresholds['quantum_advantage_threshold']
                
                if quantum_win_rate > 0.7:
                    # Quantum performing well, lower threshold (more willing to use quantum)
                    new_threshold = max(0.4, old_threshold * 0.95)  # Lower by 5%, min 0.4
                    self.routing_thresholds['quantum_advantage_threshold'] = new_threshold
                    logger.info(f"Quantum win rate {quantum_win_rate:.1%} high, "
                               f"lowered threshold {old_threshold:.2f} → {new_threshold:.2f}")
                elif quantum_win_rate < 0.4:
                    # Quantum underperforming, raise threshold (more conservative)
                    new_threshold = min(0.8, old_threshold * 1.05)  # Raise by 5%, max 0.8
                    self.routing_thresholds['quantum_advantage_threshold'] = new_threshold
                    logger.info(f"Quantum win rate {quantum_win_rate:.1%} low, "
                               f"raised threshold {old_threshold:.2f} → {new_threshold:.2f}")
        
        # =====================================================================
        # Adjust min_problem_size
        # =====================================================================
        # If small quantum problems succeed frequently, lower minimum size
        
        small_quantum = [e for e in quantum_executions 
                        if e.problem_size < self.routing_thresholds['min_problem_size']
                        and e.success]
        
        if len(small_quantum) >= 10:
            small_success_rate = sum(1 for e in small_quantum if e.success) / len(small_quantum)
            
            if small_success_rate > 0.8:
                # Quantum works well for small problems, lower threshold
                old_min = self.routing_thresholds['min_problem_size']
                new_min = max(5, old_min - 2)  # Lower by 2, minimum 5
                self.routing_thresholds['min_problem_size'] = new_min
                logger.info(f"Small quantum success rate {small_success_rate:.1%}, "
                           f"lowered min_problem_size {old_min} → {new_min}")
        
        # =====================================================================
        # Adjust max_problem_size
        # =====================================================================
        # If large quantum problems succeed, raise maximum size
        
        large_quantum = [e for e in quantum_executions
                        if e.problem_size > self.routing_thresholds['max_problem_size'] * 0.9
                        and e.success]
        
        if len(large_quantum) >= 5:
            large_success_rate = sum(1 for e in large_quantum if e.success) / len(large_quantum)
            
            if large_success_rate > 0.7:
                # Quantum handling large problems well, raise threshold
                old_max = self.routing_thresholds['max_problem_size']
                new_max = min(150, old_max + 10)  # Raise by 10, maximum 150
                self.routing_thresholds['max_problem_size'] = new_max
                logger.info(f"Large quantum success rate {large_success_rate:.1%}, "
                           f"raised max_problem_size {old_max} → {new_max}")
        
        # Log final thresholds
        logger.info(f"Updated thresholds: {self.routing_thresholds}")
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _get_size_range(self, problem_size: int) -> Tuple[int, int]:
        """
        Get size range category for a problem size.
        
        Args:
            problem_size: Number of variables/nodes
        
        Returns:
            (min, max) tuple defining the size range category
        """
        if problem_size <= 20:
            return (1, 20)  # Small
        elif problem_size <= 50:
            return (21, 50)  # Medium
        elif problem_size <= 100:
            return (51, 100)  # Large
        else:
            return (101, 999)  # Very Large
    
    def _compute_statistics(
        self,
        problem_type: str,
        size_range: Tuple[int, int],
        solver_type: str
    ) -> Optional[PerformanceStatistics]:
        """
        Compute performance statistics from execution history.
        
        Args:
            problem_type: Problem type to filter
            size_range: (min, max) size range to filter
            solver_type: Solver type to filter (classical or quantum)
        
        Returns:
            PerformanceStatistics if sufficient data, None otherwise
        """
        # Filter executions matching criteria
        matching = [
            e for e in self.execution_history
            if e.problem_type == problem_type
            and size_range[0] <= e.problem_size <= size_range[1]
            and e.solver_used == solver_type
            and e.actual_time_s is not None
            and e.actual_energy_mj is not None
        ]
        
        if len(matching) < 3:
            # Need at least 3 samples for meaningful statistics
            return None
        
        # Extract measurements
        times = [e.actual_time_s for e in matching]
        energies = [e.actual_energy_mj for e in matching]
        qualities = [e.solution_quality for e in matching if e.solution_quality is not None]
        
        # Calculate time prediction errors
        time_errors = [abs(e.actual_time_s - e.estimated_time_s) for e in matching]
        energy_errors = [abs(e.actual_energy_mj - e.estimated_energy_mj) for e in matching]
        
        # Calculate statistics
        stats = PerformanceStatistics(
            problem_type=problem_type,
            size_range=size_range,
            solver_type=solver_type,
            num_executions=len(matching),
            
            avg_time_s=statistics.mean(times),
            std_time_s=statistics.stdev(times) if len(times) > 1 else 0.0,
            avg_energy_mj=statistics.mean(energies),
            std_energy_mj=statistics.stdev(energies) if len(energies) > 1 else 0.0,
            avg_quality=statistics.mean(qualities) if qualities else 0.0,
            
            success_rate=sum(1 for e in matching if e.success) / len(matching),
            avg_prediction_error_time=statistics.mean(time_errors),
            avg_prediction_error_energy=statistics.mean(energy_errors),
        )
        
        # Calculate average quantum advantage if this is quantum solver
        if solver_type == 'quantum':
            # Compare to classical executions in same category
            classical_times = [
                e.actual_time_s for e in self.execution_history
                if e.problem_type == problem_type
                and size_range[0] <= e.problem_size <= size_range[1]
                and e.solver_used == 'classical'
                and e.actual_time_s is not None
            ]
            
            if classical_times:
                avg_classical_time = statistics.mean(classical_times)
                stats.avg_quantum_advantage = avg_classical_time / stats.avg_time_s if stats.avg_time_s > 0 else 1.0
        
        return stats
    
    def _estimate_classical_requirements(
        self,
        analysis: Dict[str, Any],
        edge_env: EdgeEnvironment
    ) -> JobRequirements:
        """
        Estimate resource requirements for classical solver.
        
        Classical solvers typically need:
        - Moderate power (CPU processing)
        - Memory for problem representation and search state
        - Storage for intermediate results
        - Execution time based on problem size
        
        Args:
            analysis: Problem analysis from ProblemAnalyzer
            edge_env: Edge environment for context
        
        Returns:
            JobRequirements for classical solver
        """
        
        problem_size = analysis['problem_size']
        runtime_s = analysis['estimated_classical_runtime']
        
        # Power: CPU processing, typically 10-30W depending on problem
        # Smaller problems use less power (idle waiting), larger use more (full utilization)
        if problem_size < 20:
            power_watts = 10.0
        elif problem_size < 50:
            power_watts = 20.0
        else:
            power_watts = 30.0
        
        # Memory: Problem representation + search state
        # QUBO matrix: O(n²) floats, plus search state
        memory_mb = max(100, int(problem_size ** 1.5))  # Superlinear growth
        
        # Storage: Input, output, intermediate results
        storage_gb = 0.1 + (problem_size / 1000)  # Minimal storage needed
        
        # Thermal output: Roughly proportional to power
        # Modern CPUs are efficient, ~80% of power becomes heat
        thermal_output_watts = power_watts * 0.8
        
        # Bandwidth: Minimal (local processing)
        bandwidth_mbps = 1.0
        
        return JobRequirements(
            power_watts=power_watts,
            execution_time_seconds=runtime_s,
            memory_mb=memory_mb,
            storage_gb=storage_gb,
            thermal_output_watts=thermal_output_watts,
            bandwidth_mbps=bandwidth_mbps
        )
    
    def _estimate_quantum_requirements(
        self,
        analysis: Dict[str, Any],
        edge_env: EdgeEnvironment
    ) -> JobRequirements:
        """
        Estimate resource requirements for quantum solver (simulator).
        
        Quantum simulators need:
        - Higher power (exponential state space simulation)
        - Exponential memory (2^n state vectors)
        - More execution time (circuit compilation + simulation)
        - Storage for circuits and results
        
        For real quantum hardware (future):
        - Lower power (Rotonium: <50W total)
        - Lower memory (only circuit description, not state)
        - Faster execution (no simulation overhead)
        
        Args:
            analysis: Problem analysis from ProblemAnalyzer
            edge_env: Edge environment for context
        
        Returns:
            JobRequirements for quantum solver
        """
        
        problem_size = analysis['problem_size']
        runtime_s = analysis['estimated_quantum_runtime']
        
        # Power: Quantum simulation is computationally intensive
        # Rotonium QPU: <50W, but simulation needs classical computer
        # For real hardware, this would be much lower (~30-50W total)
        if problem_size < 20:
            power_watts = 30.0  # Small circuits, manageable simulation
        elif problem_size < 30:
            power_watts = 40.0  # Medium circuits, more simulation load
        else:
            power_watts = 50.0  # Large circuits, heavy simulation
        
        # Memory: State vector simulation needs 2^n complex numbers
        # Each complex number: 16 bytes (2 doubles)
        # Total: 16 * 2^n bytes = 16 * 2^n / (1024^2) MB
        # For n=20: 16MB, n=25: 512MB, n=30: 16GB
        # Add overhead for circuit representation and classical processing
        state_vector_mb = int((16 * (2 ** problem_size)) / (1024 ** 2))
        overhead_mb = 500  # Circuit, gates, measurements
        memory_mb = state_vector_mb + overhead_mb
        
        # Storage: Circuit files, calibration, results
        storage_gb = 0.5 + (problem_size / 100)
        
        # Thermal output: Similar to power (efficient quantum simulation)
        # Rotonium's room-temp operation eliminates cryogenic cooling heat!
        thermal_output_watts = power_watts * 0.85
        
        # Bandwidth: May need to fetch circuits or upload results
        bandwidth_mbps = 5.0
        
        return JobRequirements(
            power_watts=power_watts,
            execution_time_seconds=runtime_s,
            memory_mb=memory_mb,
            storage_gb=storage_gb,
            thermal_output_watts=thermal_output_watts,
            bandwidth_mbps=bandwidth_mbps
        )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("QuantumRouter - Intelligent Solver Selection System")
    print("=" * 80)
    print()
    
    # Import necessary modules
    from src.problems.maxcut import MaxCutProblem
    from src.router.edge_simulator import DeploymentProfile
    
    # Create router
    router = QuantumRouter()
    
    # Test with different problem sizes and environments
    test_configs = [
        {'size': 5, 'density': 0.3, 'profile': DeploymentProfile.MOBILE},
        {'size': 25, 'density': 0.3, 'profile': DeploymentProfile.AEROSPACE},
        {'size': 50, 'density': 0.2, 'profile': DeploymentProfile.GROUND_SERVER},
        {'size': 120, 'density': 0.1, 'profile': DeploymentProfile.GROUND_SERVER},
    ]
    
    for config in test_configs:
        print(f"\nTest Case: size={config['size']}, density={config['density']}, "
              f"profile={config['profile'].value}")
        print("-" * 80)
        
        # Create problem
        problem = MaxCutProblem(num_nodes=config['size'])
        problem.generate(edge_probability=config['density'])
        
        # Create environment
        env = EdgeEnvironment(config['profile'])
        
        # Create preferences
        prefs = RoutingPreferences(
            prefer_speed=True,
            prefer_energy=(config['profile'] != DeploymentProfile.GROUND_SERVER),
            allow_hybrid=True
        )
        
        # Route problem
        result = router.route_problem(problem, env, prefs)
        
        # Print summary
        print(f"Decision: {result['decision'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Estimated Time: {result['estimated_time_ms']}ms")
        print(f"Estimated Energy: {result['estimated_energy_mj']:.2f}mJ")
        print(f"\nReasoning: {result['reasoning'][:150]}...")
        
        # Print alternatives
        if result['alternative_options']:
            print(f"\nAlternatives: {len(result['alternative_options'])} options")
            for alt in result['alternative_options'][:2]:  # Show first 2
                print(f"  - {alt['option']}: {alt['note'][:60]}...")
    
    print("\n" + "=" * 80)
    print("QuantumRouter enables intelligent quantum-classical routing for edge devices!")
    print("Rotonium's room-temperature QPU makes quantum computing practical in")
    print("resource-constrained environments (aerospace, mobile, ground server).")
    print("=" * 80)
