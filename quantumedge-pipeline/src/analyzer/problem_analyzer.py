"""Problem analyzer for characterizing optimization problems."""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ProblemComplexity(Enum):
    """Problem complexity classification."""
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ProblemStructure(Enum):
    """Problem structure type."""
    SPARSE = "sparse"
    DENSE = "dense"
    BANDED = "banded"
    BLOCK_DIAGONAL = "block_diagonal"
    RANDOM = "random"


@dataclass
class ProblemCharacteristics:
    """Characteristics of an optimization problem."""
    
    # Basic properties
    problem_type: str
    size: int
    num_variables: int
    num_constraints: Optional[int] = None
    
    # Complexity metrics
    complexity: ProblemComplexity = ProblemComplexity.MEDIUM
    structure: ProblemStructure = ProblemStructure.RANDOM
    sparsity: float = 0.5
    
    # Quantum suitability
    quantum_advantage_score: float = 0.0
    qubit_requirement: int = 0
    circuit_depth_estimate: int = 0
    
    # Resource estimates
    classical_time_estimate: float = 0.0
    quantum_time_estimate: float = 0.0
    memory_requirement_mb: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProblemAnalyzer:
    """
    Analyzes optimization problems to determine their characteristics
    and suitability for different solver types.
    """
    
    def __init__(self):
        """Initialize the problem analyzer."""
        self.complexity_thresholds = {
            "trivial": 10,
            "low": 50,
            "medium": 200,
            "high": 1000,
        }
    
    def analyze(self, problem: Any) -> ProblemCharacteristics:
        """
        Analyze a problem and return its characteristics.
        
        Args:
            problem: The optimization problem to analyze
            
        Returns:
            ProblemCharacteristics object with analysis results
        """
        # Extract basic properties
        problem_type = self._get_problem_type(problem)
        size = self._get_problem_size(problem)
        num_variables = self._get_num_variables(problem)
        num_constraints = self._get_num_constraints(problem)
        
        # Analyze complexity
        complexity = self._analyze_complexity(size, num_variables)
        structure = self._analyze_structure(problem)
        sparsity = self._calculate_sparsity(problem)
        
        # Quantum suitability analysis
        quantum_score = self._calculate_quantum_advantage(
            problem_type, size, structure, complexity
        )
        qubit_req = self._estimate_qubit_requirement(num_variables)
        circuit_depth = self._estimate_circuit_depth(num_variables, structure)
        
        # Resource estimation
        classical_time = self._estimate_classical_time(size, complexity)
        quantum_time = self._estimate_quantum_time(qubit_req, circuit_depth)
        memory_req = self._estimate_memory_requirement(size, num_variables)
        
        return ProblemCharacteristics(
            problem_type=problem_type,
            size=size,
            num_variables=num_variables,
            num_constraints=num_constraints,
            complexity=complexity,
            structure=structure,
            sparsity=sparsity,
            quantum_advantage_score=quantum_score,
            qubit_requirement=qubit_req,
            circuit_depth_estimate=circuit_depth,
            classical_time_estimate=classical_time,
            quantum_time_estimate=quantum_time,
            memory_requirement_mb=memory_req,
            metadata=self._extract_metadata(problem)
        )
    
    def _get_problem_type(self, problem: Any) -> str:
        """Extract problem type."""
        return getattr(problem, 'problem_type', 'unknown')
    
    def _get_problem_size(self, problem: Any) -> int:
        """Calculate problem size."""
        return getattr(problem, 'size', 0)
    
    def _get_num_variables(self, problem: Any) -> int:
        """Get number of decision variables."""
        return getattr(problem, 'num_variables', 0)
    
    def _get_num_constraints(self, problem: Any) -> Optional[int]:
        """Get number of constraints."""
        return getattr(problem, 'num_constraints', None)
    
    def _analyze_complexity(self, size: int, num_variables: int) -> ProblemComplexity:
        """Determine problem complexity."""
        if size <= self.complexity_thresholds["trivial"]:
            return ProblemComplexity.TRIVIAL
        elif size <= self.complexity_thresholds["low"]:
            return ProblemComplexity.LOW
        elif size <= self.complexity_thresholds["medium"]:
            return ProblemComplexity.MEDIUM
        elif size <= self.complexity_thresholds["high"]:
            return ProblemComplexity.HIGH
        else:
            return ProblemComplexity.EXTREME
    
    def _analyze_structure(self, problem: Any) -> ProblemStructure:
        """Analyze problem structure."""
        # Placeholder: analyze problem matrix/graph structure
        return ProblemStructure.RANDOM
    
    def _calculate_sparsity(self, problem: Any) -> float:
        """Calculate problem sparsity (0=dense, 1=sparse)."""
        # Placeholder: calculate actual sparsity
        return 0.5
    
    def _calculate_quantum_advantage(
        self,
        problem_type: str,
        size: int,
        structure: ProblemStructure,
        complexity: ProblemComplexity
    ) -> float:
        """
        Calculate quantum advantage score (0-1).
        Higher scores indicate better quantum suitability.
        """
        score = 0.0
        
        # Problem type factor
        quantum_friendly_types = ["maxcut", "tsp", "portfolio", "graph_coloring"]
        if problem_type.lower() in quantum_friendly_types:
            score += 0.3
        
        # Size factor (quantum advantage for medium-large problems)
        if 20 <= size <= 500:
            score += 0.3
        elif size > 500:
            score += 0.1
        
        # Complexity factor
        if complexity in [ProblemComplexity.MEDIUM, ProblemComplexity.HIGH]:
            score += 0.2
        
        # Structure factor
        if structure == ProblemStructure.SPARSE:
            score += 0.2
        
        return min(score, 1.0)
    
    def _estimate_qubit_requirement(self, num_variables: int) -> int:
        """Estimate number of qubits needed."""
        # Basic estimate: 1 qubit per variable + ancilla qubits
        return num_variables + max(1, int(np.log2(num_variables)))
    
    def _estimate_circuit_depth(self, num_variables: int, structure: ProblemStructure) -> int:
        """Estimate quantum circuit depth."""
        base_depth = num_variables * 2
        
        if structure == ProblemStructure.SPARSE:
            return int(base_depth * 0.7)
        elif structure == ProblemStructure.DENSE:
            return int(base_depth * 1.5)
        
        return base_depth
    
    def _estimate_classical_time(self, size: int, complexity: ProblemComplexity) -> float:
        """Estimate classical solver time (seconds)."""
        complexity_factors = {
            ProblemComplexity.TRIVIAL: 0.001,
            ProblemComplexity.LOW: 0.01,
            ProblemComplexity.MEDIUM: 0.1,
            ProblemComplexity.HIGH: 1.0,
            ProblemComplexity.EXTREME: 10.0,
        }
        
        factor = complexity_factors.get(complexity, 1.0)
        return size * factor * 0.001
    
    def _estimate_quantum_time(self, qubits: int, circuit_depth: int) -> float:
        """Estimate quantum solver time (seconds)."""
        # Simplified estimation including compilation and execution
        shots = 1000
        gate_time = 0.0001  # 100 microseconds per gate
        
        execution_time = circuit_depth * gate_time * shots
        compilation_overhead = 2.0  # seconds
        
        return execution_time + compilation_overhead
    
    def _estimate_memory_requirement(self, size: int, num_variables: int) -> float:
        """Estimate memory requirement (MB)."""
        # Rough estimation based on problem size
        bytes_per_variable = 8  # float64
        overhead_factor = 2.0
        
        memory_bytes = size * num_variables * bytes_per_variable * overhead_factor
        return memory_bytes / (1024 * 1024)
    
    def _extract_metadata(self, problem: Any) -> Dict[str, Any]:
        """Extract additional problem metadata."""
        metadata = {}
        
        # Extract any custom attributes
        for attr in ['description', 'source', 'tags']:
            if hasattr(problem, attr):
                metadata[attr] = getattr(problem, attr)
        
        return metadata
