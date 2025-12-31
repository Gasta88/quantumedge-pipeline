"""
Quantum Simulator for Optimization Problems in QuantumEdge Pipeline.

This module provides a quantum simulator implementation using Pennylane that
simulates photonic quantum computing characteristics. It's designed to mirror
the behavior of Rotonium's room-temperature photonic quantum computers.

Key Features:
    - Pennylane-based quantum circuit simulation
    - Photonic-specific noise modeling
    - Room-temperature noise characteristics
    - QAOA implementation for optimization
    - Energy and performance tracking

Why Simulate Photonic Quantum Computers?
    Photonic quantum computers offer unique advantages:
    1. Room temperature operation (no cryogenics needed)
    2. Lower operational energy requirements
    3. Natural encoding using Orbital Angular Momentum (OAM)
    4. Different noise characteristics than superconducting qubits
    
    By simulating these characteristics, we can:
    - Predict real hardware performance
    - Optimize algorithms for photonic systems
    - Make informed routing decisions
    - Compare against classical solvers fairly

References:
    - Pennylane: https://pennylane.ai/
    - QAOA: https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html
    - Photonic QC: https://arxiv.org/abs/2004.04375

Author: QuantumEdge Team
Created: 2025-12-29
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from abc import ABC
import time

# Scipy for classical optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning(
        "Scipy not installed. Classical optimization will not be available. "
        "Install with: pip install scipy"
    )

# Import base solver
from .solver_base import SolverBase, SolverException, SolverTimeoutError

# Pennylane imports
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logging.warning(
        "Pennylane not installed. Quantum simulation will not be available. "
        "Install with: pip install pennylane"
    )

logger = logging.getLogger(__name__)


class QuantumSimulatorException(SolverException):
    """Exception raised when quantum simulation fails."""
    pass


class CircuitDepthExceededError(QuantumSimulatorException):
    """Exception raised when circuit depth exceeds reasonable limits."""
    pass


class QuantumSimulator(SolverBase):
    """
    Quantum simulator for optimization problems using Pennylane.
    
    This class simulates quantum computing approaches to optimization problems,
    with specific focus on photonic quantum computing characteristics. It uses
    Pennylane's quantum machine learning library to construct and execute
    quantum circuits.
    
    Architecture:
        ┌─────────────────────────────────────────┐
        │      QuantumSimulator (Pennylane)       │
        ├─────────────────────────────────────────┤
        │ • Backend configuration                 │
        │ • Noise modeling (photonic-specific)    │
        │ • Circuit construction                  │
        │ • QAOA implementation                   │
        │ • Energy tracking                       │
        └─────────────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │         Quantum Circuit                 │
        │  |ψ⟩ = Σ α_i |i⟩  (superposition)       │
        │                                         │
        │  Gates: H, RX, RY, CNOT, etc.          │
        │  Measurement: computational basis       │
        └─────────────────────────────────────────┘
    
    Photonic Quantum Computing Characteristics:
        Unlike superconducting qubits (which require ~20mK temperatures),
        photonic qubits operate at room temperature using photons:
        
        Advantages:
            • Room temperature operation
            • Lower decoherence rates for certain operations
            • Natural integration with fiber optics
            • Scalability potential
            
        Challenges:
            • Photon loss (0.1-1% per operation)
            • Detection efficiency (~95-99%)
            • Gate fidelities (95-99.9% depending on operation)
            • Limited two-qubit gate repertoire
    
    Attributes:
        backend (str): Pennylane device backend (e.g., 'default.qubit')
        shots (int): Number of circuit executions per measurement
        noise_model (Optional[Dict]): Photonic noise parameters
        max_circuit_depth (int): Maximum allowed circuit depth
        device: Pennylane quantum device instance
        
    Example:
        >>> from src.problems.maxcut import MaxCutProblem
        >>> problem = MaxCutProblem(num_nodes=4, edge_probability=0.6)
        >>> problem.generate()
        >>> 
        >>> simulator = QuantumSimulator(
        ...     backend='default.qubit',
        ...     shots=1024
        ... )
        >>> 
        >>> result = simulator.solve(
        ...     problem,
        ...     algorithm='qaoa',
        ...     num_layers=3
        ... )
        >>> 
        >>> print(f"Best solution: {result['solution']}")
        >>> print(f"Energy: {result['energy_mj']:.2f} mJ")
        >>> print(f"Quantum advantage score: {result['metadata']['quantum_score']}")
    """
    
    # Default noise parameters for photonic quantum computers
    # Based on current state-of-the-art photonic systems
    DEFAULT_PHOTONIC_NOISE = {
        'photon_loss_rate': 0.005,      # 0.5% loss per operation (typical)
        'detection_efficiency': 0.97,    # 97% detection success
        'single_qubit_gate_fidelity': 0.999,  # 99.9% for single-qubit gates
        'two_qubit_gate_fidelity': 0.95,      # 95% for two-qubit gates
        'measurement_fidelity': 0.98,         # 98% measurement accuracy
        'thermal_photon_rate': 0.001,         # Room temp thermal photons
    }
    
    # Maximum circuit depth recommendations
    # Deeper circuits accumulate more noise
    MAX_CIRCUIT_DEPTH = {
        'shallow': 50,      # Quick, less accurate
        'medium': 200,      # Balanced
        'deep': 500,        # Slow, more accurate (if noise is low)
        'extreme': 1000,    # Research/testing only
    }
    
    def __init__(
        self,
        backend: str = 'default.qubit',
        shots: int = 1024,
        noise_model: Optional[Dict[str, float]] = None,
        max_circuit_depth: int = 200,
        device_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the quantum simulator with Pennylane configuration.
        
        This constructor sets up the quantum simulation environment, configures
        the backend device, and initializes noise modeling parameters specific
        to photonic quantum computing.
        
        Args:
            backend (str): Pennylane device backend identifier
                Options:
                    - 'default.qubit': Fast CPU simulator (noiseless)
                    - 'default.mixed': Density matrix simulator (supports noise)
                    - 'lightning.qubit': GPU-accelerated simulator
                    - 'qiskit.aer': Use Qiskit Aer backend
                Default: 'default.qubit'
                
            shots (int): Number of times to execute each circuit
                More shots = better statistics but slower execution
                Typical values:
                    - 100: Quick testing
                    - 1024: Standard (good balance)
                    - 10000: High precision
                Default: 1024
                
            noise_model (Optional[Dict[str, float]]): Custom noise parameters
                If None, uses DEFAULT_PHOTONIC_NOISE
                Keys:
                    - 'photon_loss_rate': Probability of photon loss per gate
                    - 'detection_efficiency': Measurement success probability
                    - 'single_qubit_gate_fidelity': Single-qubit gate accuracy
                    - 'two_qubit_gate_fidelity': Two-qubit gate accuracy
                    - 'measurement_fidelity': Measurement accuracy
                    - 'thermal_photon_rate': Thermal noise rate
                    
            max_circuit_depth (int): Maximum allowed circuit depth
                Prevents excessive computation and noise accumulation
                Typical: 50-500 depending on problem and noise
                Default: 200
                
            device_kwargs (Optional[Dict]): Additional device configuration
                Passed directly to qml.device()
                Examples: {'wires': 20}, {'analytic': True}
        
        Raises:
            SolverConfigurationError: If Pennylane not installed or invalid config
            
        Note:
            For noiseless simulation (default.qubit), noise_model is stored
            but not applied during simulation. Use 'default.mixed' for
            actual noise simulation.
        """
        # Initialize parent SolverBase
        super().__init__(
            solver_type='quantum',
            solver_name='quantum_simulator_pennylane'
        )
        
        # Check if Pennylane is available
        if not PENNYLANE_AVAILABLE:
            raise SolverConfigurationError(
                "Pennylane is not installed. Install with: pip install pennylane"
            )
        
        # Validate configuration
        if shots < 1:
            raise SolverConfigurationError(
                f"shots must be positive integer, got {shots}"
            )
        
        if max_circuit_depth < 1:
            raise SolverConfigurationError(
                f"max_circuit_depth must be positive, got {max_circuit_depth}"
            )
        
        # Store configuration
        self.backend = backend
        self.shots = shots
        self.max_circuit_depth = max_circuit_depth
        self.device_kwargs = device_kwargs or {}
        
        # Configure photonic-specific noise model
        if noise_model is None:
            self.noise_model = self.DEFAULT_PHOTONIC_NOISE.copy()
        else:
            # Start with defaults and override with user values
            self.noise_model = self.DEFAULT_PHOTONIC_NOISE.copy()
            self.noise_model.update(noise_model)
        
        # Apply photonic-specific configuration
        self._configure_photonic_backend()
        
        # Device will be created when needed (lazy initialization)
        # This allows us to adjust qubit count based on problem size
        self.device = None
        self._current_num_qubits = 0
        
        logger.info(
            f"QuantumSimulator initialized: backend={backend}, "
            f"shots={shots}, max_depth={max_circuit_depth}"
        )
        logger.debug(f"Noise model: {self.noise_model}")
    
    def _configure_photonic_backend(self) -> None:
        """
        Configure simulation parameters specific to photonic quantum computers.
        
        This method sets up noise characteristics that mirror real photonic
        quantum computing systems. Photonic systems have different noise
        profiles compared to superconducting or ion trap systems.
        
        Photonic Noise Characteristics:
        
        1. Photon Loss:
           - Occurs during transmission and gate operations
           - Typical rates: 0.1-1% per operation
           - Wavelength dependent (lower loss at telecom wavelengths)
           - Formula: P_loss = 1 - exp(-α * L)
             where α is loss coefficient, L is length
        
        2. Detection Efficiency:
           - Single Photon Detectors (SPDs) not perfect
           - State-of-art: 95-99% efficiency
           - Dark counts: ~10-100 Hz (room temperature)
        
        3. Gate Fidelities:
           - Single-qubit gates: 99.9% (phase shifts, rotations)
           - Two-qubit gates: 95-98% (more complex, uses interference)
           - Limited gate set compared to superconducting
        
        4. Thermal Photons:
           - Room temperature advantage
           - Thermal photon rate: ~0.001 at telecom wavelengths
           - Much lower than superconducting thermal excitation
        
        5. Decoherence:
           - Photons don't decohere during flight (advantage!)
           - Loss is the main error mechanism
           - No T1/T2 relaxation like superconducting qubits
        
        Why This Matters:
            - Accurate simulation helps predict real hardware performance
            - Different noise profiles favor different algorithms
            - Informs routing decisions (quantum vs classical)
            - Helps set expectations for hybrid approaches
        
        Configuration Adjustments:
            - For noiseless backends: Store parameters for analysis
            - For noisy backends: Apply actual noise channels
            - Adjust max circuit depth based on noise rates
        
        Reference:
            Slussarenko & Pryde, "Photonic quantum information processing: 
            A concise review" (2019), Applied Physics Reviews 6, 041303
        """
        # Calculate effective circuit depth limit based on noise
        # Higher noise = shallower circuits recommended
        photon_loss = self.noise_model['photon_loss_rate']
        two_qubit_fidelity = self.noise_model['two_qubit_gate_fidelity']
        
        # Estimate maximum useful depth before noise dominates
        # Rule of thumb: when error probability > 50%, circuit is unreliable
        # For photonic systems, photon loss is the dominant error
        
        # Photon survival probability after n operations
        # P_survive(n) = (1 - loss_rate)^n
        # We want P_survive > 0.5 for useful computation
        # Solving: (1 - loss_rate)^n > 0.5
        # n < log(0.5) / log(1 - loss_rate)
        
        if photon_loss > 0:
            theoretical_max_depth = int(
                np.log(0.5) / np.log(1 - photon_loss)
            )
            
            # Be conservative: use 80% of theoretical max
            recommended_max_depth = int(0.8 * theoretical_max_depth)
            
            if self.max_circuit_depth > recommended_max_depth:
                logger.warning(
                    f"Circuit depth {self.max_circuit_depth} exceeds recommended "
                    f"max {recommended_max_depth} for photon loss rate {photon_loss:.4f}. "
                    f"Results may be unreliable due to noise accumulation."
                )
        
        # Log photonic configuration
        logger.info(
            f"Photonic backend configured:\n"
            f"  Photon loss: {photon_loss*100:.2f}%/operation\n"
            f"  Detection efficiency: {self.noise_model['detection_efficiency']*100:.1f}%\n"
            f"  Two-qubit fidelity: {two_qubit_fidelity*100:.1f}%\n"
            f"  Operating temperature: Room temperature (~300K)\n"
            f"  Max recommended depth: {self.max_circuit_depth}"
        )
        
        # Additional backend-specific configuration
        if self.backend == 'default.mixed':
            # Mixed state simulator supports noise channels
            logger.info("Using mixed-state simulator for noise modeling")
            # Note: Actual noise channels would be applied in circuit construction
            
        elif self.backend == 'default.qubit':
            # Pure state simulator - noiseless
            logger.info(
                "Using pure-state simulator (noiseless). "
                "Noise parameters stored for analysis only."
            )
        
        else:
            logger.info(f"Using backend: {self.backend}")
    
    def _validate_circuit_depth(self, depth: int, num_qubits: int) -> None:
        """
        Validate that circuit depth is within reasonable limits.
        
        Circuit depth is a critical parameter in quantum computing:
        - Shallow circuits: Fast, less noise, but may be less accurate
        - Deep circuits: More expressive, but accumulate more noise
        
        For photonic systems, depth limits are determined by:
        1. Photon loss accumulation
        2. Gate error accumulation  
        3. Computational time constraints
        
        This method checks if the requested circuit depth is reasonable
        given the noise model and prevents excessive computation.
        
        Depth Accumulation in QAOA:
            For QAOA with p layers:
            - Each layer has: 2 * num_edges + num_qubits gates
            - Total depth ≈ p * (2 * num_edges + num_qubits)
            - Typical: p=1-10, num_qubits=10-100
            - Results in depths: 50-5000
        
        Noise Impact:
            - Single-qubit error per layer: ε_1 * num_qubits
            - Two-qubit error per layer: ε_2 * num_edges
            - Total error ≈ p * (ε_1 * n + ε_2 * m)
            
            For photonic systems with our default noise:
            - ε_1 ≈ 0.001 (single-qubit)
            - ε_2 ≈ 0.05 (two-qubit)
            
            Example: p=3, n=20, m=40 (dense graph)
            - Error ≈ 3 * (0.001*20 + 0.05*40) = 3 * 2.02 = 6.06
            - Photon survival ≈ (1-0.005)^(3*60) ≈ 0.54 (54%)
        
        Args:
            depth (int): Proposed circuit depth
            num_qubits (int): Number of qubits in circuit
            
        Raises:
            CircuitDepthExceededError: If depth exceeds maximum
            
        Note:
            This is a soft limit for warning purposes. Actual execution
            may still proceed with user override, but results may be
            unreliable for very deep circuits.
        """
        if depth > self.max_circuit_depth:
            # Calculate expected photon survival
            loss_rate = self.noise_model['photon_loss_rate']
            survival_prob = (1 - loss_rate) ** depth
            
            error_msg = (
                f"Circuit depth {depth} exceeds maximum {self.max_circuit_depth}.\n"
                f"With photon loss rate {loss_rate:.4f}, expected photon "
                f"survival: {survival_prob*100:.1f}%\n"
                f"Deep circuits accumulate noise and may produce unreliable results.\n"
                f"Consider:\n"
                f"  1. Using fewer QAOA layers (p)\n"
                f"  2. Reducing problem size\n"
                f"  3. Using classical solver instead\n"
                f"  4. Increasing max_circuit_depth (at your own risk)"
            )
            
            raise CircuitDepthExceededError(error_msg)
        
        # Warn if depth is substantial
        if depth > self.max_circuit_depth * 0.7:
            logger.warning(
                f"Circuit depth {depth} is approaching maximum {self.max_circuit_depth}. "
                f"Results may be affected by noise accumulation."
            )
        
        # Estimate execution time
        # Rough estimate: ~0.1-1ms per gate on modern simulators
        # For shots, multiply by shot count
        estimated_gates = depth * num_qubits  # Very rough estimate
        estimated_time_ms = estimated_gates * 0.5 * self.shots / 1000
        
        if estimated_time_ms > 60000:  # > 1 minute
            logger.warning(
                f"Circuit execution may take ~{estimated_time_ms/1000:.0f} seconds "
                f"with {self.shots} shots. Consider reducing shots or circuit depth."
            )
    
    def _create_device(self, num_qubits: int) -> Any:
        """
        Create or update Pennylane device with specified number of qubits.
        
        Lazy device initialization allows us to adjust qubit count based
        on problem size without recreating the simulator for each problem.
        
        Args:
            num_qubits (int): Number of qubits needed
            
        Returns:
            Pennylane device instance
        """
        # Only recreate device if qubit count changed
        if self.device is None or num_qubits != self._current_num_qubits:
            logger.debug(f"Creating Pennylane device: {self.backend} with {num_qubits} qubits")
            
            self.device = qml.device(
                self.backend,
                wires=num_qubits,
                shots=self.shots,
                **self.device_kwargs
            )
            self._current_num_qubits = num_qubits
        
        return self.device
    
    def _qubo_to_hamiltonian(self, qubo_matrix: np.ndarray) -> Any:
        """
        Convert QUBO matrix to Pennylane Hamiltonian.
        
        QUBO (Quadratic Unconstrained Binary Optimization) problems have the form:
            minimize: x^T Q x
            where x ∈ {0, 1}^n (binary variables)
            and Q is the QUBO matrix
        
        Mathematical Conversion:
        -----------------------
        
        1. Binary to Spin Mapping:
           QUBO uses binary variables x_i ∈ {0, 1}
           Quantum computers use spin variables σ_i ∈ {-1, +1}
           
           Mapping: x_i = (1 - σ_i) / 2
           
           Where σ_i is represented by Pauli-Z operator:
           - |0⟩ state → σ_i = +1 → x_i = 0
           - |1⟩ state → σ_i = -1 → x_i = 1
        
        2. QUBO to Ising Hamiltonian:
           Starting with QUBO: Σ_ij Q_ij x_i x_j
           
           Substitute x_i = (1 - σ_i) / 2:
           = Σ_ij Q_ij [(1 - σ_i) / 2] [(1 - σ_j) / 2]
           = Σ_ij Q_ij [(1 - σ_i - σ_j + σ_i σ_j) / 4]
           
           Expanding terms:
           = (1/4) Σ_ij Q_ij [1 - σ_i - σ_j + σ_i σ_j]
           
           Grouping by order:
           - Constant: (1/4) Σ_ij Q_ij
           - Linear: -(1/4) Σ_ij Q_ij (σ_i + σ_j)
           - Quadratic: (1/4) Σ_ij Q_ij σ_i σ_j
        
        3. Pennylane Hamiltonian Representation:
           In Pennylane, we represent this using Pauli operators:
           
           H = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + constant
           
           Where:
           - Z_i is Pauli-Z operator on qubit i
           - h_i are local field coefficients
           - J_ij are coupling coefficients
        
        4. Coefficient Calculation:
           For diagonal QUBO terms Q_ii:
               h_i = -Q_ii / 2 - (1/2) Σ_{j≠i} Q_ij
           
           For off-diagonal QUBO terms Q_ij (i ≠ j):
               J_ij = Q_ij / 4
           
           Constant term (can be ignored for optimization):
               C = (1/4) Σ_ij Q_ij
        
        Why We Need This:
        ------------------
        - QAOA works with quantum operators (Hamiltonians)
        - QUBO is a classical formulation
        - This conversion allows us to encode classical optimization
          problems into quantum circuits
        - The eigenvalues of the Hamiltonian correspond to QUBO objective values
        
        Example:
        --------
        For MaxCut on 2 nodes with edge weight 1:
            QUBO = [[-1,  1],
                    [ 1, -1]]
            
        This becomes:
            H = -0.5 * Z_0 - 0.5 * Z_1 + 0.25 * Z_0 Z_1
            
        Where measuring |01⟩ or |10⟩ gives lower energy (better cut)
        than |00⟩ or |11⟩ (no cut).
        
        Args:
            qubo_matrix (np.ndarray): n×n QUBO coefficient matrix
                                      Typically symmetric for optimization problems
        
        Returns:
            qml.Hamiltonian: Pennylane Hamiltonian object ready for QAOA
        
        Note:
            The constant term is typically omitted as it doesn't affect
            the optimization (doesn't change relative ordering of solutions).
        """
        n = qubo_matrix.shape[0]  # Number of qubits needed
        
        # Lists to store Hamiltonian terms
        coeffs = []  # Coefficients for each term
        obs = []     # Observables (Pauli operators) for each term
        
        # Process diagonal terms (local fields)
        # These represent individual variable contributions
        for i in range(n):
            # Calculate local field coefficient
            # h_i includes diagonal term and half of all interactions with i
            h_i = -qubo_matrix[i, i] / 2.0
            
            # Add contributions from interactions (off-diagonal terms)
            for j in range(n):
                if i != j:
                    h_i -= qubo_matrix[i, j] / 2.0
            
            # Only add non-zero terms (efficiency)
            if abs(h_i) > 1e-10:
                coeffs.append(h_i)
                obs.append(qml.PauliZ(i))  # Z operator on qubit i
        
        # Process off-diagonal terms (couplings)
        # These represent interactions between pairs of variables
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle (avoid double-counting)
                # Calculate coupling coefficient
                # Average Q_ij and Q_ji in case matrix is not perfectly symmetric
                J_ij = (qubo_matrix[i, j] + qubo_matrix[j, i]) / 4.0
                
                # Only add non-zero couplings
                if abs(J_ij) > 1e-10:
                    coeffs.append(J_ij)
                    # Z_i ⊗ Z_j: measures correlation between qubits i and j
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        # Construct Pennylane Hamiltonian
        # H = Σ coeffs[k] * obs[k]
        if len(coeffs) == 0:
            # Edge case: all-zero QUBO (trivial problem)
            logger.warning("QUBO matrix is all zeros, creating identity Hamiltonian")
            coeffs = [0.0]
            obs = [qml.Identity(0)]
        
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        
        logger.debug(
            f"Converted QUBO to Hamiltonian: {len(coeffs)} terms "
            f"({n} qubits, {sum(1 for o in obs if len(o.wires) == 1)} local fields, "
            f"{sum(1 for o in obs if len(o.wires) == 2)} couplings)"
        )
        
        return hamiltonian
    
    def _build_qaoa_circuit(
        self,
        qubo_matrix: np.ndarray,
        params: np.ndarray,
        p: int = 1
    ) -> callable:
        """
        Build QAOA (Quantum Approximate Optimization Algorithm) circuit.
        
        QAOA Overview:
        --------------
        QAOA is a hybrid quantum-classical algorithm designed to find approximate
        solutions to combinatorial optimization problems. It was introduced by
        Farhi et al. (2014) and is one of the most promising near-term quantum
        algorithms (NISQ era).
        
        How QAOA Works:
        ---------------
        
        1. Problem Encoding:
           - Encode optimization problem as a Hamiltonian H_C (cost Hamiltonian)
           - The ground state of H_C encodes the optimal solution
           - Energy eigenvalues correspond to objective function values
        
        2. Quantum State Preparation:
           - Start with uniform superposition: |ψ_0⟩ = (1/√2^n) Σ|x⟩
           - This is achieved by applying Hadamard gates to all qubits
           - Represents equal probability of all possible solutions
           - This is our initial "guess" - completely random!
        
        3. Variational Circuit (Ansatz):
           QAOA circuit has alternating layers (repeated p times):
           
           a) Problem Layer - encode cost function:
              U_C(γ) = exp(-i γ H_C)
              
              - Applies phase rotations based on problem structure
              - γ (gamma) is a variational parameter (angle)
              - Implements "time evolution" under H_C
              - Encodes problem structure into quantum state
              - For QUBO: involves Z and ZZ rotations
           
           b) Mixer Layer - enable exploration:
              U_M(β) = exp(-i β H_M)
              
              - H_M is typically the X mixer: Σ X_i
              - β (beta) is a variational parameter (angle)
              - Creates superpositions to explore solution space
              - Prevents getting stuck in local optima
              - Analogous to "hopping" in simulated annealing
           
           The full circuit with p layers:
           |ψ(γ, β)⟩ = U_M(β_p) U_C(γ_p) ... U_M(β_1) U_C(γ_1) |+⟩^n
        
        4. Measurement:
           - Measure all qubits in computational basis
           - Each measurement gives a candidate solution
           - Measurement probabilities reflect solution quality
           - Better solutions have higher probability amplitudes
        
        5. Classical Optimization:
           - Measure expectation value ⟨H_C⟩ = cost function
           - Use classical optimizer to adjust angles γ, β
           - Goal: minimize ⟨H_C⟩ → find better solutions
           - Common optimizers: COBYLA, ADAM, L-BFGS-B
           - Repeat circuit + measurement + optimization
        
        Role of Parameters (Angles):
        ----------------------------
        
        Gamma (γ) - Problem Angles:
        - Control how much problem structure to encode
        - Small γ → weak encoding, state stays in superposition
        - Large γ → strong encoding, state moves toward low-energy configurations
        - Optimal γ depends on problem instance and layer number
        
        Beta (β) - Mixer Angles:
        - Control exploration vs exploitation trade-off
        - Small β → stay close to current state (exploitation)
        - Large β → broad exploration of solution space
        - β ≈ π/4 often works well for X mixer
        
        Why Use Superposition:
        ----------------------
        - Classical algorithms check solutions one at a time (or a few)
        - Quantum superposition allows checking many solutions simultaneously
        - Quantum interference amplifies good solutions, suppresses bad ones
        - This is the source of potential quantum advantage!
        
        But:
        - We don't get all solutions instantly (measurement collapses state)
        - Need many measurements to see the probability distribution
        - Classical optimization guides us to better parameter settings
        
        Circuit Depth and Layers:
        -------------------------
        
        p = 1 (Shallow):
        - Fast execution, minimal noise accumulation
        - Can find reasonable solutions for simple problems
        - May miss optimal solution for complex problems
        - Circuit depth: O(n) for sparse problems, O(n²) for dense
        
        p > 1 (Deep):
        - More expressive, can approximate optimal solution better
        - Higher circuit depth → more noise in real hardware
        - More parameters to optimize (2p parameters total)
        - Theoretical guarantee: as p→∞, can reach exact solution
        
        Rule of thumb: Start with p=1, increase if solution quality insufficient
        
        Circuit Structure (for this implementation):
        --------------------------------------------
        
        Step 1: Initialize in superposition
            H|0⟩^n → |+⟩^n = (1/√2^n) Σ|x⟩
            
        Step 2: For each layer l = 1 to p:
            a) Apply cost Hamiltonian evolution:
               - For each Z term (local field): RZ(2*γ_l*h_i) on qubit i
               - For each ZZ term (coupling): exp(-i*γ_l*J_ij*Z_i*Z_j)
                 Implemented as: CNOT-RZ-CNOT sequence
               
            b) Apply mixer Hamiltonian evolution:
               - For each qubit: RX(2*β_l) (X rotation)
               - This is X mixer: exp(-i*β*Σ X_i)
        
        Step 3: Measure all qubits
        
        Performance Characteristics:
        ---------------------------
        - Time complexity: O(p * m * shots) where m is # of Hamiltonian terms
        - Space complexity: O(2^n) for state vector simulation
        - Shot count: More shots → better statistics → slower but more accurate
        - Parameter count: 2p (can be optimized with ~10-1000 iterations)
        
        Args:
            qubo_matrix (np.ndarray): QUBO matrix defining the problem
            params (np.ndarray): Variational parameters [γ_1, β_1, ..., γ_p, β_p]
                                 Shape: (2*p,) where p is number of layers
            p (int): Number of QAOA layers (depth)
                     Default: 1 (shallowest QAOA)
                     Typical range: 1-10
        
        Returns:
            callable: Pennylane QNode (quantum circuit function)
                      Can be called with params to execute circuit
        
        Example:
            >>> qubo = np.array([[-1, 1], [1, -1]])  # MaxCut on 2 nodes
            >>> params = np.array([0.5, 0.3])  # [γ, β] for p=1
            >>> circuit = self._build_qaoa_circuit(qubo, params, p=1)
            >>> expectation = circuit(params)  # Execute and measure
        
        References:
            - Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
            - Pennylane QAOA tutorial: https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html
        """
        n = qubo_matrix.shape[0]  # Number of qubits
        
        # Convert QUBO to quantum Hamiltonian
        hamiltonian = self._qubo_to_hamiltonian(qubo_matrix)
        
        # Create or get device for this problem size
        device = self._create_device(n)
        
        # Extract Hamiltonian components for circuit construction
        # hamiltonian.terms() returns (coefficients, observables)
        h_coeffs, h_ops = hamiltonian.terms()
        
        # Validate parameters
        expected_params = 2 * p
        if len(params) != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters for p={p} layers, "
                f"got {len(params)}"
            )
        
        # Define the quantum circuit as a Pennylane QNode
        @qml.qnode(device)
        def circuit(params):
            """
            The actual QAOA quantum circuit.
            
            This function defines the quantum operations that will be executed.
            It's decorated with @qml.qnode which tells Pennylane to compile it
            into an executable quantum circuit on the specified device.
            
            Args:
                params: Variational parameters [γ_1, β_1, ..., γ_p, β_p]
            
            Returns:
                Expectation value of the cost Hamiltonian ⟨ψ|H_C|ψ⟩
            """
            
            # ============================================================
            # STEP 1: Initialize all qubits in equal superposition
            # ============================================================
            # Apply Hadamard gate to each qubit: H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
            # 
            # Why superposition?
            # - Classical computers start with one specific state (e.g., all zeros)
            # - Quantum computers can start in ALL possible states simultaneously
            # - For n qubits: |+⟩^n = (1/√2^n) Σ|x⟩ over all 2^n bitstrings x
            # - This is quantum parallelism - we're "trying all solutions at once"
            # 
            # Mathematical notation:
            # |+⟩^n = H^⊗n |0⟩^n
            #       = ⊗_{i=1}^n (|0⟩_i + |1⟩_i)/√2
            #       = (1/√2^n) Σ_{x∈{0,1}^n} |x⟩
            #
            for i in range(n):
                qml.Hadamard(wires=i)
            
            # ============================================================
            # STEP 2: Apply p layers of QAOA evolution
            # ============================================================
            # Each layer has two components:
            # 1. Cost layer: Encodes problem structure (uses γ angles)
            # 2. Mixer layer: Explores solution space (uses β angles)
            #
            for layer in range(p):
                # Extract parameters for this layer
                gamma = params[2 * layer]      # Cost angle for this layer
                beta = params[2 * layer + 1]   # Mixer angle for this layer
                
                # --------------------------------------------------------
                # STEP 2a: Apply Cost Hamiltonian U_C(γ) = exp(-i γ H_C)
                # --------------------------------------------------------
                # This encodes the optimization problem into the quantum state
                # 
                # For each term in the Hamiltonian, we apply a rotation:
                # - Single Z terms (h_i * Z_i): Apply RZ gate
                # - Double ZZ terms (J_ij * Z_i Z_j): Apply CNOT-RZ-CNOT
                #
                # The angle γ controls how strongly we encode the problem:
                # - Small γ: weak encoding, stay mostly in superposition
                # - Large γ: strong encoding, move toward low-energy states
                #
                # Mathematical form:
                # exp(-i γ h_i Z_i) = RZ(2γ h_i)  [single qubit rotation]
                # exp(-i γ J_ij Z_i Z_j)          [two qubit rotation]
                #
                for coeff, op in zip(h_coeffs, h_ops):
                    # Get qubits involved in this term
                    qubits = op.wires.tolist()
                    
                    if len(qubits) == 1:
                        # Single-qubit term: local field
                        # Apply RZ rotation: RZ(θ) = exp(-i θ Z/2)
                        # We want exp(-i γ coeff Z), so θ = 2*γ*coeff
                        qml.RZ(2 * gamma * coeff, wires=qubits[0])
                        
                    elif len(qubits) == 2:
                        # Two-qubit term: coupling between variables
                        # We want to apply: exp(-i γ J_ij Z_i Z_j)
                        # 
                        # This is implemented using the identity:
                        # exp(-i θ Z_i Z_j) = CNOT(i,j) RZ(2θ) CNOT(i,j)
                        #
                        # Circuit:
                        #   q_i: ─────●────────────●─────
                        #             │            │
                        #   q_j: ─────X───RZ(2θ)───X─────
                        #
                        q_i, q_j = qubits
                        angle = 2 * gamma * coeff
                        
                        qml.CNOT(wires=[q_i, q_j])
                        qml.RZ(angle, wires=q_j)
                        qml.CNOT(wires=[q_i, q_j])
                
                # --------------------------------------------------------
                # STEP 2b: Apply Mixer Hamiltonian U_M(β) = exp(-i β H_M)
                # --------------------------------------------------------
                # The mixer drives transitions between different states
                # 
                # Standard choice: X mixer, H_M = Σ_i X_i
                # Applies X rotations to all qubits
                #
                # Why X mixer?
                # - X operator flips qubits: X|0⟩ = |1⟩, X|1⟩ = |0⟩
                # - RX(β) creates superposition of current state and flipped state
                # - This allows algorithm to "explore" nearby solutions
                # - Without mixer, we'd be stuck in initial superposition
                #
                # The angle β controls exploration:
                # - Small β: minor perturbations, exploitation
                # - Large β: major changes, exploration
                # - β ≈ π: maximum mixing (flip all qubits completely)
                #
                # Mathematical form:
                # exp(-i β X_i) = RX(2β)
                # RX(θ) = cos(θ/2)I - i*sin(θ/2)X
                #
                for i in range(n):
                    qml.RX(2 * beta, wires=i)
            
            # ============================================================
            # STEP 3: Measure expectation value of cost Hamiltonian
            # ============================================================
            # We want to know: ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩
            # 
            # This expectation value is the cost function we're minimizing
            # 
            # How measurement works:
            # 1. Quantum state after QAOA: |ψ(γ,β)⟩ = Σ α_x |x⟩
            #    where α_x are complex amplitudes
            # 
            # 2. Measurement probabilities: P(x) = |α_x|²
            #    When we measure, we get bitstring x with probability P(x)
            # 
            # 3. Expectation value calculation:
            #    ⟨H⟩ = Σ_x P(x) * E(x)
            #    where E(x) is the energy (cost) of bitstring x
            # 
            # 4. Pennylane automatically:
            #    - Runs circuit multiple times (shots)
            #    - Collects measurement statistics
            #    - Computes expectation value
            #    - Returns scalar value for optimizer
            #
            # Note: We're measuring the cost Hamiltonian, not individual qubits
            # Pennylane handles this through operator expectation values
            #
            return qml.expval(hamiltonian)
        
        # Validate circuit depth
        # Estimate depth: ~2 gates per Hamiltonian term per layer + mixer gates
        estimated_depth = p * (len(h_coeffs) * 3 + n)  # Conservative estimate
        self._validate_circuit_depth(estimated_depth, n)
        
        logger.debug(
            f"Built QAOA circuit: {n} qubits, {p} layers, "
            f"{len(params)} parameters, ~{estimated_depth} gates"
        )
        
        return circuit
    
    def _measure_qaoa_expectation(
        self,
        circuit: callable,
        params: np.ndarray
    ) -> float:
        """
        Execute QAOA circuit and measure expectation value.
        
        This function is the bridge between the quantum circuit and the
        classical optimization loop. It:
        1. Executes the quantum circuit with given parameters
        2. Performs measurements (shots)
        3. Computes the expectation value of the cost Hamiltonian
        4. Returns this value to the classical optimizer
        
        What is Expectation Value?
        --------------------------
        The expectation value ⟨H⟩ is the average energy/cost we would get
        if we measured the quantum state many times:
        
        ⟨H⟩ = ⟨ψ|H|ψ⟩ = Σ_x P(x) * E(x)
        
        Where:
        - |ψ⟩ is our quantum state after QAOA circuit
        - P(x) = |⟨x|ψ⟩|² is probability of measuring bitstring x
        - E(x) is the energy (objective value) of solution x
        
        Example:
        --------
        Suppose we have a MaxCut problem and after QAOA we get:
        
        |ψ⟩ = 0.6|01⟩ + 0.6|10⟩ + 0.4|00⟩ + 0.4|11⟩
        
        Measurement probabilities:
        - P(01) = 0.36  →  E(01) = -1 (good cut)
        - P(10) = 0.36  →  E(10) = -1 (good cut)
        - P(00) = 0.16  →  E(00) = +1 (bad cut)
        - P(11) = 0.16  →  E(11) = +1 (bad cut)
        
        Expectation value:
        ⟨H⟩ = 0.36*(-1) + 0.36*(-1) + 0.16*(+1) + 0.16*(+1)
            = -0.36 - 0.36 + 0.16 + 0.16
            = -0.40
        
        Lower (more negative) is better!
        Good QAOA parameters concentrate amplitude on good solutions.
        
        Why This is the Cost Function:
        ------------------------------
        In the outer classical optimization loop, we're trying to find
        the best parameters (γ, β) that minimize ⟨H⟩.
        
        Optimization process:
        1. Start with random or heuristic parameters
        2. Measure ⟨H⟩ with current parameters (this function)
        3. Classical optimizer adjusts parameters to reduce ⟨H⟩
        4. Repeat until convergence
        
        The optimizer is essentially doing gradient descent (or similar)
        in parameter space to find the angles that give lowest energy.
        
        Measurement Statistics:
        ----------------------
        With finite shots, we get statistical estimates:
        
        - True expectation: ⟨H⟩_true
        - Measured estimate: ⟨H⟩_measured ≈ ⟨H⟩_true
        - Uncertainty: σ ~ 1/√shots
        
        More shots → better estimate → slower but more accurate
        
        Trade-offs:
        - 100 shots: Fast, noisy gradient signals, may converge poorly
        - 1000 shots: Balanced (typical choice)
        - 10000 shots: Slow, smooth optimization, better convergence
        
        Noise Effects:
        -------------
        In real hardware or noisy simulation:
        - Photon loss: Some qubits measured wrong
        - Gate errors: Circuit doesn't implement exact evolution
        - Measurement errors: Wrong bitstrings recorded
        
        All of these add noise to ⟨H⟩, making optimization harder.
        More shots can help average out noise, but systematic errors remain.
        
        Photonic Specifics:
        ------------------
        For photonic quantum computers:
        - Detection efficiency: ~97% → 3% of measurements invalid
        - Photon loss: ~0.5% per gate → accumulated over circuit
        - These errors bias the expectation value
        - Typically make ⟨H⟩ closer to zero (less negative)
        - Can prevent finding true optimal solution
        
        Implementation Details:
        ----------------------
        Pennylane automatically handles:
        1. Circuit compilation to device-specific gates
        2. Shot-based sampling of computational basis
        3. Expectation value calculation from samples
        4. Gradient computation (if using autodiff)
        
        We just call circuit(params) and get back ⟨H⟩!
        
        Args:
            circuit (callable): Pennylane QNode (compiled quantum circuit)
                               Created by _build_qaoa_circuit()
            params (np.ndarray): Variational parameters [γ_1, β_1, ..., γ_p, β_p]
        
        Returns:
            float: Expectation value ⟨H⟩ = ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩
                  This is the cost function value to be minimized
                  Lower values indicate better solutions
        
        Note:
            - This function is called many times during optimization
            - Each call executes the full circuit 'shots' times
            - Total quantum circuit executions = shots * optimization_steps
            - For shots=1024 and 100 optimization steps: 102,400 circuit runs!
        
        Example:
            >>> params = np.array([0.5, 0.3])  # [γ, β] for p=1
            >>> cost = self._measure_qaoa_expectation(circuit, params)
            >>> print(f"Current cost: {cost:.4f}")
            Current cost: -0.8234
        """
        # Execute the quantum circuit with given parameters
        # Pennylane will:
        # 1. Compile the circuit to device-specific gates
        # 2. Run the circuit 'self.shots' times
        # 3. Collect measurement outcomes
        # 4. Calculate expectation value from measurement statistics
        # 5. Return the scalar expectation value
        
        try:
            expectation = circuit(params)
            
            # Log execution for debugging (can be verbose in optimization loop)
            logger.debug(
                f"QAOA expectation: {expectation:.6f} "
                f"(params: {params})"
            )
            
            return float(expectation)
            
        except Exception as e:
            # Circuit execution can fail for various reasons:
            # - Device errors
            # - Memory limits
            # - Invalid parameters
            # - Backend issues
            logger.error(f"Circuit execution failed: {e}")
            raise QuantumSimulatorException(
                f"Failed to measure QAOA expectation: {e}"
            )
    
    def _apply_photonic_noise(
        self,
        circuit_func: callable,
        num_qubits: int,
        num_gates: int
    ) -> float:
        """
        Apply realistic photonic noise to circuit execution.
        
        Photonic Quantum Computing: Noise Characteristics
        ==================================================
        
        Unlike superconducting qubits (which operate at ~20 millikelvin and have
        T1/T2 coherence times of ~100 microseconds), photonic quantum computers
        have fundamentally different noise profiles:
        
        Photonic Noise Sources:
        -----------------------
        
        1. **Photon Loss** (Dominant Error Source):
           - Occurs during waveguide transmission
           - Beam splitters and phase shifters introduce loss
           - Coupling between components has inefficiency
           - Detection has <100% efficiency
           
           Typical rates:
           - Propagation loss: 0.1-1 dB/cm (wavelength dependent)
           - Per-gate loss: 0.5-2% per operation
           - Fiber coupling: 1-5% per connection
           - Detection loss: 1-5% (even with good detectors)
           
           Why it matters:
           - Lost photons → missing information
           - Can't recover lost photons (unlike superconducting bit flips)
           - Accumulates multiplicatively: P_survive = (1-loss)^n_gates
           
        2. **Mode Mismatch** (Imperfect Interference):
           - Photonic gates rely on quantum interference
           - Requires photons in identical spatial/temporal modes
           - Imperfect mode matching → reduced gate fidelity
           
           Effects:
           - Two-photon gates most sensitive
           - Hong-Ou-Mandel interference not perfect
           - Visibility ~95-99% for good systems
           
           Causes:
           - Manufacturing tolerances in waveguides
           - Temperature fluctuations (even at room temp)
           - Wavelength instabilities
           - Spatial mode distortions
        
        3. **Detector Inefficiency**:
           - Single Photon Detectors (SPDs) not 100% efficient
           - State-of-art: 95-99% quantum efficiency
           - Every measurement has chance of missing photon
           
           Types of detectors:
           - Superconducting Nanowire SPDs: 95-99%, need cooling
           - Avalanche Photodiodes (APD): 50-70%, room temp
           - Transition Edge Sensors: >99%, need cooling
           
           Impact:
           - Lost measurements → statistical bias
           - Need post-selection (throw out failed detections)
           - Reduces effective measurement rate
        
        Why Photonic Noise Differs from Superconducting:
        ------------------------------------------------
        
        Superconducting Qubits:
        ❌ Thermal excitation (need extreme cooling)
        ❌ Energy relaxation (T1 ~ 50-100 μs)
        ❌ Dephasing (T2 ~ 50-100 μs)
        ❌ Crosstalk between qubits
        ✓ High gate fidelities (99.9% single, 99% two-qubit)
        
        Photonic Qubits:
        ✓ No thermal excitation (photons at room temp)
        ✓ No decoherence during flight (photons don't interact with environment)
        ✓ No crosstalk (photons in separate modes don't interfere)
        ❌ Photon loss (main error mechanism)
        ❌ Detector inefficiency
        ❌ Mode mismatch
        
        Room Temperature Operation:
        ---------------------------
        
        Why photonic systems can work at room temperature:
        
        1. **Photon Energy Scale**:
           - Telecom photons (1550 nm): E = hν ≈ 0.8 eV
           - Thermal energy at 300K: kT ≈ 0.026 eV
           - Ratio: E/(kT) ≈ 30
           - Thermal excitation: exp(-E/kT) ≈ 10^-13 (negligible!)
        
        2. **No Material Excitations**:
           - Photons don't live in a material (unlike superconducting qubits)
           - No substrate to thermally excite
           - Waveguides have bandgap >> kT
           - Only concern is detector dark counts
        
        3. **Decoherence-Free Subspace**:
           - Photons in vacuum don't decohere
           - Path encoding is robust
           - Polarization stable over long distances
           - Time-bin encoding immune to slow drifts
        
        This is HUGE for edge deployment:
        ✓ No cryogenic infrastructure (saves space, weight, power)
        ✓ Easier maintenance (no helium, no vibration isolation)
        ✓ Lower operating cost (~90% reduction vs cryogenic)
        ✓ Faster startup (no cooldown time)
        ✓ More reliable (fewer moving parts)
        
        Noise Simulation Implementation:
        --------------------------------
        
        We model photonic noise by adjusting the expectation value:
        
        1. Photon loss per gate:
           P_survive = (1 - loss_rate)^num_gates
           Measured value reduced by survival probability
        
        2. Mode mismatch:
           Effective gate fidelity reduced
           Two-qubit gate fidelity: 95-98% (vs 99% for superconducting)
           Adds random noise to measurement
        
        3. Detector inefficiency:
           Fraction of measurements lost
           Increases shot noise in expectation value
        
        Args:
            circuit_func (callable): Quantum circuit function to execute
            num_qubits (int): Number of qubits in circuit
            num_gates (int): Approximate number of gates in circuit
        
        Returns:
            float: Expectation value with photonic noise applied
        
        Note:
            This is a simplified noise model. Real photonic systems have more
            complex noise correlations, but this captures the dominant effects
            for QAOA simulations.
        
        References:
            - Slussarenko & Pryde, "Photonic quantum information processing" (2019)
            - Qiang et al., "Large-scale silicon quantum photonics" (2018)
            - Madsen et al., "Quantum computational advantage with photons" (2022)
        """
        # Extract photonic noise parameters
        photon_loss_rate = self.noise_model['photon_loss_rate']
        detection_efficiency = self.noise_model['detection_efficiency']
        two_qubit_fidelity = self.noise_model['two_qubit_gate_fidelity']
        measurement_fidelity = self.noise_model['measurement_fidelity']
        
        # 1. Model photon loss through circuit
        # Photons can be lost at any gate, accumulates multiplicatively
        photon_survival_prob = (1 - photon_loss_rate) ** num_gates
        
        # 2. Model mode mismatch (gate imperfections)
        # Estimate number of two-qubit gates (rough: ~30% of total gates)
        num_two_qubit_gates = int(num_gates * 0.3)
        # Each imperfect two-qubit gate adds noise
        mode_mismatch_factor = two_qubit_fidelity ** num_two_qubit_gates
        
        # 3. Model detector inefficiency
        # Effective measurement rate reduced
        effective_detection = detection_efficiency * measurement_fidelity
        
        # Combined noise factor
        # These multiply because they are independent error sources
        total_noise_factor = (
            photon_survival_prob *      # Photon loss
            mode_mismatch_factor *      # Gate imperfections
            effective_detection         # Detection errors
        )
        
        # Execute the circuit
        expectation = circuit_func()
        
        # Apply noise model to expectation value
        # Lost photons and detector inefficiency reduce signal strength
        noisy_expectation = expectation * total_noise_factor
        
        # Add measurement shot noise (photon counting statistics)
        # Standard error in expectation value: σ ~ 1/√shots
        shot_noise_std = 1.0 / np.sqrt(self.shots)
        noise = np.random.normal(0, shot_noise_std)
        noisy_expectation += noise
        
        logger.debug(
            f"Photonic noise applied: "
            f"Survival={photon_survival_prob:.4f}, "
            f"Fidelity={mode_mismatch_factor:.4f}, "
            f"Detection={effective_detection:.4f}, "
            f"Total factor={total_noise_factor:.4f}"
        )
        
        return noisy_expectation
    
    def _simulate_room_temperature_effects(
        self,
        circuit_depth: int,
        num_qubits: int
    ) -> Dict[str, float]:
        """
        Model room temperature operation effects for photonic quantum computers.
        
        Room Temperature Quantum Computing: The Photonic Advantage
        ===========================================================
        
        One of the most significant advantages of photonic quantum computing is
        the ability to operate at room temperature (~300K) rather than requiring
        cryogenic cooling to ~20 millikelvin like superconducting qubits.
        
        Why Room Temperature Operation is Possible:
        -------------------------------------------
        
        The Physics:
        ------------
        
        1. **Energy Gap vs Thermal Energy**:
        
           For a quantum system to be stable against thermal excitations:
               E_gap >> k_B * T
           
           Superconducting qubits:
               - Energy gap: ~200 MHz = 4 μeV
               - At 300K: k_B*T = 26 meV
               - Ratio: E_gap / (k_B*T) ≈ 0.00015 << 1
               - Result: DESTROYED by thermal noise at room temp!
               - Solution: Cool to 20 mK where k_B*T = 2 μeV
        
           Photonic qubits (1550 nm telecom wavelength):
               - Energy: E = hν = 0.8 eV = 800,000 μeV
               - At 300K: k_B*T = 26 meV = 26,000 μeV
               - Ratio: E / (k_B*T) ≈ 30 >> 1
               - Thermal excitation prob: exp(-30) ≈ 10^-13
               - Result: Essentially ZERO thermal excitation!
        
        2. **Decoherence Mechanisms**:
        
           Superconducting qubits:
               - Store information in material excitations
               - Material interacts with environment
               - Phonons, two-level systems, magnetic field noise
               - T1 (energy relaxation): 50-100 μs
               - T2 (dephasing): 50-100 μs
               - Worse at higher temperatures
        
           Photonic qubits:
               - Information encoded in photon state (polarization, path, time)
               - Photons don't interact with environment (bosons!)
               - No decoherence during propagation in vacuum/fiber
               - T1, T2 → ∞ for photons in flight
               - Only decohere at measurement
        
        3. **Isolation from Environment**:
        
           Superconducting qubits:
               - Embedded in chip substrate
               - Capacitively/inductively coupled to environment
               - Sensitive to electromagnetic noise
               - Need extensive shielding and filtering
        
           Photonic qubits:
               - Confined to waveguides (glass/silicon)
               - Minimal interaction with substrate
               - Immune to electric/magnetic fields
               - Natural isolation
        
        Practical Advantages for Edge Computing:
        ----------------------------------------
        
        1. **No Cryogenic Infrastructure**:
           ❌ No dilution refrigerator ($500K+)
           ❌ No liquid helium supply
           ❌ No vibration isolation tables
           ❌ No vacuum pumps
           ✓ Standard electronics cooling (air/liquid)
        
        2. **Energy Consumption**:
           Superconducting system:
               - Dilution fridge: ~10-20 kW continuous
               - Control electronics: ~1 kW
               - Total: ~11-21 kW for ~50 qubits
               - Efficiency: ~200-400 W per qubit!
        
           Photonic system:
               - Laser sources: ~100-500 mW per channel
               - Detectors: ~10-100 mW each
               - Control electronics: ~100 W
               - Total: ~1-2 W for ~50 qubits
               - Efficiency: ~20-40 mW per qubit!
               - **~10,000× more energy efficient!**
        
        3. **Form Factor**:
           Superconducting:
               - Refrigerator: 2m × 2m × 3m
               - Weight: ~1000 kg
               - Needs lab space
        
           Photonic:
               - Chip: 1cm × 1cm × 0.5mm
               - With packaging: 10cm × 10cm × 5cm
               - Weight: <1 kg
               - **Portable!**
        
        4. **Deployment Time**:
           Superconducting:
               - Cooldown: 24-48 hours
               - Warmup for maintenance: 24-48 hours
               - Downtime for repairs: days
        
           Photonic:
               - Startup: <1 minute (laser stabilization)
               - Maintenance: hot-swappable components
               - Downtime: minutes
               - **>1000× faster deployment**
        
        Trade-offs (Why Everyone Doesn't Use Photonic):
        -----------------------------------------------
        
        Challenges:
        ❌ Photon loss (0.5-2% per gate vs 0.1% superconducting)
        ❌ Two-qubit gates harder (need Hong-Ou-Mandel interference)
        ❌ Deterministic photon sources challenging
        ❌ Scaling to many qubits requires many photons
        ❌ No long-term quantum memory (photons always moving)
        
        Advantages:
        ✓ Room temperature operation
        ✓ No decoherence during computation
        ✓ Natural networking (fiber optics)
        ✓ Lower energy consumption
        ✓ Smaller form factor
        ✓ Faster deployment
        
        For Edge Computing Specifically:
        --------------------------------
        
        Edge devices need:
        ✓ Low power (battery operated)
        ✓ Small size (embedded systems)
        ✓ Reliable (minimal maintenance)
        ✓ Fast startup (on-demand computation)
        ✓ Cost-effective (commodity hardware)
        
        Photonic QPU wins on ALL these metrics!
        
        Coherence Time Modeling:
        ------------------------
        
        Effective coherence times for photonic systems:
        
        1. During propagation: T_coherence → ∞
           (photons don't decohere in vacuum/waveguide)
        
        2. During storage (if needed): T_storage ~ 1 μs - 1 ms
           (limited by cavity finesse, absorption)
        
        3. Effective coherence for QAOA:
           - Circuit execution time: ~1-10 μs per gate
           - Total circuit time: circuit_depth × 1 μs
           - Loss-limited: effective_time ~ 1/(loss_rate × gate_rate)
        
        We model "effective coherence" as being limited by photon loss,
        not by decoherence. This is fundamentally different from
        superconducting systems!
        
        Args:
            circuit_depth (int): Number of gates in circuit
            num_qubits (int): Number of qubits
        
        Returns:
            Dict[str, float]: Room temperature operation characteristics
                {
                    'operating_temperature_k': 300.0,
                    'thermal_excitation_prob': ~10^-13,
                    'effective_coherence_time_us': loss-limited,
                    'circuit_execution_time_us': estimate,
                    'thermal_photon_rate': ~0.001,
                    'cooling_power_w': ~0,
                    'system_power_w': low,
                }
        
        Note:
            These are idealized values. Real systems have additional
            considerations like laser stability, temperature control
            for phase stability, etc.
        
        References:
            - O'Brien et al., "Photonic quantum technologies" (2009)
            - Rudolph, "Why I am optimistic about photonic quantum computing" (2017)
            - Arute et al., "Quantum supremacy using superconducting processor" (2019)
              (for superconducting comparison)
        """
        # Operating temperature (room temperature!)
        operating_temp_k = 300.0  # ~27°C, comfortable room temperature
        
        # Thermal photon rate at telecom wavelengths
        # Blackbody radiation at 1550 nm and 300K
        # n_thermal = 1/(exp(hν/kT) - 1) ≈ exp(-hν/kT) for hν >> kT
        wavelength_nm = 1550  # Telecom C-band
        photon_energy_ev = 1240 / wavelength_nm  # E = hc/λ in eV
        thermal_energy_ev = 8.617e-5 * operating_temp_k  # k_B*T in eV
        
        thermal_excitation_prob = np.exp(-photon_energy_ev / thermal_energy_ev)
        # Result: ~10^-13, essentially zero!
        
        # Thermal photon rate (blackbody at telecom wavelength)
        thermal_photon_rate = self.noise_model.get('thermal_photon_rate', 0.001)
        
        # Effective coherence time (limited by photon loss, not decoherence!)
        # In photonic systems, "coherence time" is really "photon survival time"
        photon_loss_rate = self.noise_model['photon_loss_rate']
        gate_time_us = 0.1  # Typical: ~100 ns per gate
        
        # Time until photon likely lost: T_loss = 1/(loss_rate * gate_rate)
        effective_coherence_time_us = 1.0 / (photon_loss_rate / gate_time_us) if photon_loss_rate > 0 else float('inf')
        
        # Circuit execution time
        circuit_execution_time_us = circuit_depth * gate_time_us
        
        # Power consumption (NO cryogenic cooling needed!)
        # Photonic system components:
        cooling_power_w = 0.0  # No cryogenic cooling!
        
        # Power breakdown:
        laser_power_w = num_qubits * 0.1  # ~100 mW per photon source
        detector_power_w = num_qubits * 0.05  # ~50 mW per detector
        control_electronics_w = 10.0  # Classical control system
        thermal_stabilization_w = 5.0  # Keep chip at constant temp (±0.1K)
        
        total_system_power_w = (
            laser_power_w +
            detector_power_w +
            control_electronics_w +
            thermal_stabilization_w
        )
        
        # Compare to superconducting (would need ~10-20 kW for dilution fridge)
        superconducting_equivalent_power_w = 15000.0  # 15 kW typical
        power_advantage_factor = superconducting_equivalent_power_w / total_system_power_w
        
        characteristics = {
            'operating_temperature_k': operating_temp_k,
            'thermal_excitation_prob': float(thermal_excitation_prob),
            'thermal_photon_rate': thermal_photon_rate,
            'effective_coherence_time_us': effective_coherence_time_us,
            'circuit_execution_time_us': circuit_execution_time_us,
            'cooling_power_w': cooling_power_w,
            'laser_power_w': laser_power_w,
            'detector_power_w': detector_power_w,
            'control_electronics_w': control_electronics_w,
            'thermal_stabilization_w': thermal_stabilization_w,
            'total_system_power_w': total_system_power_w,
            'superconducting_equivalent_power_w': superconducting_equivalent_power_w,
            'power_advantage_factor': power_advantage_factor,
            'wavelength_nm': wavelength_nm,
            'photon_energy_ev': photon_energy_ev,
        }
        
        logger.debug(
            f"Room temperature operation: "
            f"T={operating_temp_k}K, "
            f"Power={total_system_power_w:.2f}W, "
            f"Advantage={power_advantage_factor:.0f}×"
        )
        
        return characteristics
    
    def _calculate_photonic_energy_consumption(
        self,
        circuit_depth: int,
        num_qubits: int,
        num_shots: int
    ) -> Dict[str, float]:
        """
        Calculate energy consumption for photonic quantum processing unit.
        
        Energy Efficiency: Photonic vs Superconducting Quantum Computers
        =================================================================
        
        Energy consumption is CRITICAL for edge computing applications where
        devices run on batteries or have limited power budgets. Photonic quantum
        computers have a massive advantage here.
        
        Energy Budget Breakdown:
        ------------------------
        
        SUPERCONDUCTING QUANTUM COMPUTER:
        ----------------------------------
        
        1. Cryogenic Cooling (Dilution Refrigerator):
           - Target temperature: 10-20 millikelvin (0.01-0.02 K)
           - Cooling stages: 4K → 1K → 100mK → 10mK
           - Power consumption: 10-20 kW continuous
           - This is the DOMINANT cost!
        
        2. Why So Much Power for Cooling?
           - Carnot efficiency: η = T_cold / (T_hot - T_cold)
           - To remove 1 W at 20 mK: need ~15,000 W at 300K!
           - Multiple cooling stages, each inefficient
           - Heat leaks from cables, radiation
           - Pulse tube coolers, compressors
        
        3. Control Electronics:
           - Microwave generators: ~500 W
           - Arbitrary waveform generators: ~200 W
           - Digital logic: ~200 W
           - Subtotal: ~1 kW
        
        4. Total for 50-qubit system:
           - Cooling: 15 kW
           - Electronics: 1 kW
           - Total: ~16 kW continuous
           - Per qubit: ~300 W
           - Per gate: ~300 W × 1 μs = 0.3 mJ
        
        PHOTONIC QUANTUM COMPUTER:
        --------------------------
        
        1. Photon Sources (Lasers):
           - Continuous-wave lasers: ~100 mW per channel
           - Alternative: On-chip quantum dots: ~10 mW
           - For 50 qubits: 50 × 0.1 W = 5 W
        
        2. Detectors (Single Photon Detectors):
           - Superconducting nanowires: ~50 mW each (still need cooling)
           - Avalanche photodiodes: ~20 mW each (room temp!)
           - Transition edge sensors: ~10 mW each
           - For 50 qubits: 50 × 0.05 W = 2.5 W
        
        3. Chip Thermal Stabilization:
           - Optical path lengths sensitive to temperature
           - Need stability: ±0.1 K
           - Peltier cooler: ~5 W
        
        4. Control Electronics:
           - Phase modulators: ~50 W
           - Timing electronics: ~30 W
           - Computer: ~20 W
           - Subtotal: ~100 W
        
        5. Total for 50-qubit system:
           - Lasers: 5 W
           - Detectors: 2.5 W
           - Thermal: 5 W
           - Electronics: 100 W
           - Total: ~113 W continuous
           - Per qubit: ~2 W
           - Per gate: ~2 W × 0.1 μs = 0.2 μJ
        
        COMPARISON:
        -----------
        
        |                    | Superconducting | Photonic | Advantage |
        |--------------------|----------------|----------|-----------|
        | System power       | 16,000 W       | 113 W    | 140×      |
        | Per qubit          | 300 W          | 2 W      | 150×      |
        | Per gate           | 0.3 mJ         | 0.2 μJ   | 1500×     |
        | Cooling needed     | Yes (10kW+)    | No       | ∞         |
        | Startup time       | 24-48 hours    | <1 min   | 2000×     |
        | Portability        | Lab-bound      | Portable | Yes!      |
        
        For Edge Computing:
        -------------------
        
        Battery capacity example: 100 Wh (typical laptop battery)
        
        Superconducting QPU:
           - Runtime: 100 Wh / 16 kW = 0.00625 hours = 22 seconds
           - NOT VIABLE for battery operation!
        
        Photonic QPU:
           - Runtime: 100 Wh / 113 W = 0.88 hours = 53 minutes
           - VIABLE for battery operation!
        
        Solar power example: 100 W panel
        
        Superconducting QPU:
           - Would need: 160 panels (16 m² of panels)
           - NOT VIABLE for solar!
        
        Photonic QPU:
           - Would need: 1.13 panels (~1.2 m²)
           - VIABLE for solar!
        
        Energy Scaling:
        ---------------
        
        Energy per QAOA circuit execution:
        
        E_circuit = P_system × t_circuit
        
        Where:
            P_system = total system power
            t_circuit = circuit_depth × gate_time + readout_time
        
        For photonic system:
            t_gate ~ 100 ns (optical switching time)
            t_readout ~ 1 μs (detector response)
            P_lasers scales with num_qubits
            P_detectors scales with num_qubits
        
        Total energy for full QAOA solve:
            E_total = num_shots × num_iterations × E_circuit
        
        Why Photonic Scales Better:
        ----------------------------
        
        1. **Linear power scaling**:
           - Power ∝ num_qubits (one laser + detector per qubit)
           - Superconducting: cooling power increases with chip size
        
        2. **No cooling overhead**:
           - Photonic: power goes into computation
           - Superconducting: 90% power goes into cooling
        
        3. **Faster gates**:
           - Photonic: 100 ns (speed of light!)
           - Superconducting: 1 μs (RC time constants)
           - 10× faster → 10× less energy per gate
        
        4. **On-demand operation**:
           - Photonic: turn on/off instantly
           - Superconducting: must keep cold continuously
           - Idle power: photonic ~0W, superconducting ~15kW
        
        Implementation:
        ---------------
        
        We calculate energy consumption by modeling:
        1. Photon generation energy (lasers)
        2. Circuit operation energy (modulators, switches)
        3. Detection energy (single photon detectors)
        4. Control electronics energy
        5. Thermal stabilization energy
        
        Args:
            circuit_depth (int): Number of gates in circuit
            num_qubits (int): Number of qubits
            num_shots (int): Number of circuit executions
        
        Returns:
            Dict[str, float]: Energy consumption breakdown
                {
                    'total_energy_mj': total energy in millijoules,
                    'energy_per_shot_mj': energy per circuit execution,
                    'energy_per_gate_uj': energy per gate operation,
                    'laser_energy_mj': photon generation,
                    'detector_energy_mj': measurement,
                    'control_energy_mj': classical electronics,
                    'thermal_energy_mj': temperature stabilization,
                    'superconducting_equivalent_mj': what it would cost in superconducting,
                    'energy_advantage_factor': how much better than superconducting,
                }
        
        Note:
            These are estimates based on current photonic technology.
            Actual values depend on specific implementation details.
        
        References:
            - Paesani et al., "Generation and sampling of quantum states of light" (2019)
            - Pitsios et al., "Photonic simulation of entanglement growth" (2017)
            - Arute et al., "Quantum supremacy using superconducting processor" (2019)
        """
        # Time per gate operation
        gate_time_us = 0.1  # 100 nanoseconds (optical switching time)
        readout_time_us = 1.0  # 1 microsecond (detector response time)
        
        # Time for one circuit execution
        circuit_time_us = circuit_depth * gate_time_us + readout_time_us
        circuit_time_s = circuit_time_us * 1e-6
        
        # Power consumption components (in Watts)
        
        # 1. Laser power: Need to generate photons
        # Continuous-wave laser: ~100 mW per qubit
        laser_power_w = num_qubits * 0.1  # 100 mW per photon source
        
        # 2. Detector power: Single photon detectors
        # Modern APDs or SNSPDs: ~50 mW per detector
        detector_power_w = num_qubits * 0.05  # 50 mW per detector
        
        # 3. Control electronics: Phase modulators, switches, control logic
        # Electro-optic modulators: ~50 W total
        # Digital control: ~50 W
        control_power_w = 100.0
        
        # 4. Thermal stabilization: Keep chip at constant temperature
        # Peltier cooling to maintain ±0.1K stability
        thermal_power_w = 5.0
        
        # Total system power
        total_power_w = (
            laser_power_w +
            detector_power_w +
            control_power_w +
            thermal_power_w
        )
        
        # Energy for all shots
        total_time_s = num_shots * circuit_time_s
        total_energy_j = total_power_w * total_time_s
        total_energy_mj = total_energy_j * 1000  # Convert to millijoules
        
        # Energy per shot
        energy_per_shot_j = total_power_w * circuit_time_s
        energy_per_shot_mj = energy_per_shot_j * 1000
        
        # Energy per gate
        gate_time_s = gate_time_us * 1e-6
        energy_per_gate_j = total_power_w * gate_time_s
        energy_per_gate_uj = energy_per_gate_j * 1e6  # Convert to microjoules
        
        # Energy breakdown
        laser_fraction = laser_power_w / total_power_w
        detector_fraction = detector_power_w / total_power_w
        control_fraction = control_power_w / total_power_w
        thermal_fraction = thermal_power_w / total_power_w
        
        laser_energy_mj = total_energy_mj * laser_fraction
        detector_energy_mj = total_energy_mj * detector_fraction
        control_energy_mj = total_energy_mj * control_fraction
        thermal_energy_mj = total_energy_mj * thermal_fraction
        
        # Compare to superconducting system
        # Superconducting: ~16 kW for similar qubit count
        superconducting_power_w = 16000.0
        superconducting_energy_j = superconducting_power_w * total_time_s
        superconducting_energy_mj = superconducting_energy_j * 1000
        
        # Advantage factor
        energy_advantage = superconducting_energy_mj / total_energy_mj if total_energy_mj > 0 else float('inf')
        
        energy_breakdown = {
            'total_energy_mj': float(total_energy_mj),
            'energy_per_shot_mj': float(energy_per_shot_mj),
            'energy_per_gate_uj': float(energy_per_gate_uj),
            'laser_energy_mj': float(laser_energy_mj),
            'detector_energy_mj': float(detector_energy_mj),
            'control_energy_mj': float(control_energy_mj),
            'thermal_energy_mj': float(thermal_energy_mj),
            'total_power_w': float(total_power_w),
            'circuit_time_us': float(circuit_time_us),
            'total_time_s': float(total_time_s),
            'superconducting_equivalent_mj': float(superconducting_energy_mj),
            'energy_advantage_factor': float(energy_advantage),
        }
        
        logger.debug(
            f"Photonic energy: {total_energy_mj:.2f} mJ "
            f"({energy_advantage:.0f}× better than superconducting)"
        )
        
        return energy_breakdown
    
    def add_oam_encoding_advantage(
        self,
        num_qubits: int,
        oam_dimension: int = 4
    ) -> Dict[str, Any]:
        """
        Simulate benefits of Orbital Angular Momentum (OAM) encoding for photonic qubits.
        
        Orbital Angular Momentum (OAM): Beyond Binary Qubits
        =====================================================
        
        Standard quantum computing uses qubits: 2-level quantum systems (|0⟩ and |1⟩).
        But photons have additional degrees of freedom that can encode more information!
        
        Photon Encoding Options:
        ------------------------
        
        1. **Polarization** (Standard, 2 states):
           - Horizontal |H⟩ or Vertical |V⟩
           - Easy to generate and measure
           - Limited to qubits (d=2)
           - Used in most quantum communication
        
        2. **Path/Spatial Mode** (2+ states):
           - Which waveguide/path the photon takes
           - Can have many paths (d=2,3,4,...)
           - Requires many waveguides
           - Used in integrated photonic circuits
        
        3. **Time-Bin** (2+ states):
           - When the photon arrives (early/late)
           - Multiple time slots possible
           - Robust against environmental noise
           - Used in quantum key distribution
        
        4. **Orbital Angular Momentum - OAM** (MANY states!):
           - Photon's helical phase structure
           - Infinite dimensional Hilbert space!
           - Practical: d=10-100 or more
           - This is what makes OAM special!
        
        What is Orbital Angular Momentum?
        ----------------------------------
        
        Classical analogy:
        - Imagine water going down a drain - it spirals
        - Photons can have this spiral/helical wavefront
        - Different spiral patterns = different OAM states
        
        Mathematical description:
        - Electric field: E(r,φ) = E₀(r) × exp(i·ℓ·φ)
        - ℓ is the OAM quantum number (integer: ...-2,-1,0,+1,+2,...)
        - φ is azimuthal angle
        - Each ℓ is an orthogonal state!
        
        Visual (side view of wavefront):
        
        ℓ=0 (normal):     ℓ=1 (spiral):    ℓ=2 (double spiral):
        ─────────         ╱      ╲         ╱  ╲  ╱  ╲
        ─────────        │  ↻     │       │ ↻ │ ↻  │
        ─────────         ╲      ╱         ╲  ╱  ╲  ╱
        (plane wave)      (helix)          (double helix)
        
        Why OAM is Powerful:
        --------------------
        
        1. **High-Dimensional Hilbert Space**:
           - Qubits: H = C² (2 dimensions)
           - Qutrits: H = C³ (3 dimensions)
           - OAM: H = C^d where d can be 10, 50, 100+!
        
        2. **More Information Per Photon**:
           - Qubit: 1 bit per photon
           - OAM (d=8): log₂(8) = 3 bits per photon
           - OAM (d=32): log₂(32) = 5 bits per photon
           - 3-5× more information density!
        
        3. **Reduced Physical Resources**:
           - 100 qubits in polarization: need 100 photons
           - 100 "qubits" in OAM (d=32): need 100/5 = 20 photons
           - 5× fewer photons needed!
        
        4. **Compatibility with Other Encodings**:
           - Can use OAM AND polarization simultaneously
           - Hybrid qubit-qudit systems
           - Even more information per photon
        
        Advantages for Quantum Computing:
        ---------------------------------
        
        1. **Fewer Physical Qubits**:
           - A 4-dimensional qudit = 2 qubits worth of info
           - A 16-dimensional qudit = 4 qubits worth of info
           - Reduces circuit complexity
        
        2. **Native High-Dimensional Gates**:
           - Some operations easier in high-d space
           - Quantum error correction codes benefit
           - Certain algorithms more efficient
        
        3. **Longer Coherence**:
           - OAM robust against some noise sources
           - Different OAM modes don't couple easily
           - Better isolation than path encoding
        
        4. **Networking Advantages**:
           - Multiple OAM channels through same fiber
           - Wavelength-independent (can combine with WDM)
           - Higher capacity quantum links
        
        Challenges (Why Not Everyone Uses OAM):
        ---------------------------------------
        
        ❌ **Generation Complexity**:
           - Need spatial light modulators or spiral phase plates
           - Quantum dot sources challenging for high ℓ
           - Mode purity requirements strict
        
        ❌ **Detection Complexity**:
           - Need to measure which OAM mode
           - Requires mode sorters or holographic detection
           - More complex than simple polarization measurement
        
        ❌ **Gate Operations**:
           - Two-photon gates harder with OAM
           - Mode coupling must be controlled
           - Fewer demonstrated gate operations
        
        ❌ **Propagation Issues**:
           - Atmospheric turbulence scrambles OAM
           - Fiber dispersion couples modes
           - Need special fibers or free space
        
        For Edge Computing:
        -------------------
        
        OAM encoding particularly valuable for:
        
        ✓ **Resource-Constrained Devices**:
          - Fewer photons needed → less laser power
          - Fewer detectors needed → lower cost
          - More compact systems possible
        
        ✓ **Communication**:
          - Higher channel capacity for quantum networks
          - Edge devices can communicate more efficiently
          - Better use of fiber infrastructure
        
        ✓ **Specific Algorithms**:
          - Quantum simulation of high-d systems
          - Certain optimization problems
          - Quantum machine learning (features as qudits)
        
        Simplified Simulation:
        ----------------------
        
        We simulate OAM encoding by:
        
        1. **Information Density**:
           - Calculate equivalent qubit count: n_qubits_eff = n_photons × log₂(d)
           - More information with fewer physical resources
        
        2. **Resource Reduction**:
           - Fewer photons needed → less photon loss
           - Fewer gates needed → faster circuits
           - Lower energy consumption
        
        3. **Noise Model**:
           - OAM modes can couple (mode mixing)
           - Detection errors different from polarization
           - But overall more robust in some scenarios
        
        Note: Real OAM systems are MUCH more complex!
        This is a simplified model to capture the key advantages.
        
        Args:
            num_qubits (int): Number of logical qubits needed
            oam_dimension (int): Dimension of OAM encoding (d)
                               Typical: 2 (qubit), 3 (qutrit), 4, 8, 16, 32
                               Practical limit: ~100 with current technology
        
        Returns:
            Dict[str, Any]: OAM encoding advantages
                {
                    'oam_dimension': d,
                    'bits_per_photon': log₂(d),
                    'num_photons_needed': n_qubits / log₂(d),
                    'photon_reduction_factor': reduction vs standard,
                    'effective_qubit_count': logical qubits available,
                    'information_density_factor': improvement,
                    'estimated_gate_reduction': fewer gates needed,
                    'energy_savings_factor': power reduction,
                }
        
        Example:
            >>> # Compare standard polarization vs OAM encoding
            >>> # For 100 logical qubits
            >>> 
            >>> # Standard polarization (d=2):
            >>> # Need 100 photons, each carries 1 bit
            >>> 
            >>> simulator.add_oam_encoding_advantage(100, oam_dimension=2)
            >>> # Returns: {'num_photons_needed': 100, ...}
            >>> 
            >>> # OAM encoding (d=16):
            >>> # Need 25 photons, each carries 4 bits
            >>> simulator.add_oam_encoding_advantage(100, oam_dimension=16)
            >>> # Returns: {'num_photons_needed': 25, 'photon_reduction_factor': 4, ...}
        
        References:
            - Allen et al., "Orbital angular momentum of light" (1992)
            - Wang et al., "Terabit free-space data transmission employing OAM" (2012)
            - Erhard et al., "Twisted photons: new quantum perspectives" (2018)
            - Malik et al., "Multi-photon entanglement in high dimensions" (2016)
        """
        # Validate OAM dimension
        if oam_dimension < 2:
            raise ValueError("OAM dimension must be at least 2 (qubit)")
        
        if oam_dimension > 100:
            logger.warning(
                f"OAM dimension {oam_dimension} is very high. "
                f"Current technology typically supports d≤32 reliably."
            )
        
        # Information capacity per photon
        bits_per_photon = np.log2(oam_dimension)
        
        # Number of photons needed (vs qubits)
        # Each photon with dimension d carries log₂(d) qubits worth of info
        num_photons_needed = int(np.ceil(num_qubits / bits_per_photon))
        
        # Reduction factor vs standard polarization encoding
        photon_reduction_factor = num_qubits / num_photons_needed
        
        # Effective qubit count we can represent
        effective_qubit_count = num_photons_needed * bits_per_photon
        
        # Information density improvement
        information_density_factor = bits_per_photon
        
        # Estimated gate reduction
        # Fewer photons → fewer two-photon gates needed
        # Rough estimate: gates scale as O(n²) for connectivity
        standard_gate_count_estimate = num_qubits ** 1.5  # Rough scaling
        oam_gate_count_estimate = num_photons_needed ** 1.5
        gate_reduction_factor = standard_gate_count_estimate / oam_gate_count_estimate
        
        # Energy savings
        # Fewer photons → less laser power needed
        # Fewer gates → faster circuits → less time
        # Combined effect
        energy_savings_factor = photon_reduction_factor * np.sqrt(gate_reduction_factor)
        
        # OAM mode coupling (noise consideration)
        # Higher d → more modes → more coupling chances
        # But also more orthogonal states → better isolation
        mode_coupling_factor = 1.0 / np.sqrt(oam_dimension)  # Simplified model
        
        oam_advantages = {
            'oam_dimension': int(oam_dimension),
            'bits_per_photon': float(bits_per_photon),
            'num_qubits_requested': int(num_qubits),
            'num_photons_needed': int(num_photons_needed),
            'photon_reduction_factor': float(photon_reduction_factor),
            'effective_qubit_count': float(effective_qubit_count),
            'information_density_factor': float(information_density_factor),
            'estimated_gate_reduction': float(gate_reduction_factor),
            'energy_savings_factor': float(energy_savings_factor),
            'mode_coupling_factor': float(mode_coupling_factor),
            'encoding_type': 'OAM (Orbital Angular Momentum)',
            'technology_readiness': self._get_oam_trl(oam_dimension),
        }
        
        logger.info(
            f"OAM encoding (d={oam_dimension}): "
            f"{num_photons_needed} photons for {num_qubits} qubits "
            f"({photon_reduction_factor:.1f}× reduction)"
        )
        
        return oam_advantages
    
    def _get_oam_trl(self, dimension: int) -> str:
        """
        Get Technology Readiness Level for OAM encoding of given dimension.
        
        Args:
            dimension: OAM dimension
        
        Returns:
            str: TRL description
        """
        if dimension <= 2:
            return "TRL 9 - Proven technology (standard qubits)"
        elif dimension <= 4:
            return "TRL 7-8 - Demonstrated in operational environment"
        elif dimension <= 16:
            return "TRL 5-6 - Laboratory demonstration"
        elif dimension <= 32:
            return "TRL 3-4 - Proof of concept"
        else:
            return "TRL 1-2 - Basic research"
    
    def solve(
        self,
        problem: Any,
        p: int = 1,
        maxiter: int = 100,
        optimizer: str = 'COBYLA',
        initial_params: Optional[np.ndarray] = None,
        convergence_threshold: float = 1e-6,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Solve optimization problem using QAOA with hybrid quantum-classical optimization.
        
        ═══════════════════════════════════════════════════════════════════════════
        HYBRID QUANTUM-CLASSICAL OPTIMIZATION: THE KEY INSIGHT
        ═══════════════════════════════════════════════════════════════════════════
        
        QAOA is NOT purely quantum - it's a HYBRID algorithm that combines:
        
        🔵 QUANTUM PART (this runs on quantum hardware/simulator):
           - Prepare quantum state in superposition
           - Apply parameterized QAOA circuit
           - Measure expectation value of cost Hamiltonian
           - This evaluates the cost function for given parameters
        
        🟢 CLASSICAL PART (this runs on regular CPU):
           - Take expectation value from quantum circuit
           - Compute gradient or use gradient-free method
           - Adjust parameters (γ, β) to minimize cost
           - Send new parameters back to quantum circuit
           - Repeat until convergence
        
        Why Hybrid?
        -----------
        Pure quantum algorithms are hard to design and limited to specific problems.
        Hybrid algorithms leverage:
        ✓ Quantum speedup for cost function evaluation (exponential parallelism)
        ✓ Classical optimization expertise (decades of research in optimization)
        ✓ Near-term quantum devices (NISQ era - not error-corrected)
        ✓ Practical implementation on real hardware
        
        The Optimization Loop (Variational Quantum Algorithm):
        -------------------------------------------------------
        
        1. INITIALIZE: Start with random or heuristic parameters θ = [γ₁, β₁, ..., γₚ, βₚ]
        
        2. QUANTUM EVALUATION:
           |ψ(θ)⟩ ← QAOA_circuit(θ)      [Run on quantum device]
           cost ← ⟨ψ(θ)|H|ψ(θ)⟩           [Measure expectation value]
        
        3. CLASSICAL OPTIMIZATION:
           θ_new ← optimizer.step(cost, θ)  [Update parameters]
           
        4. CONVERGENCE CHECK:
           if |cost_new - cost_old| < threshold:
               DONE - parameters optimized!
           else:
               θ ← θ_new, goto step 2
        
        5. FINAL SOLUTION:
           Run optimized circuit multiple times
           Sample bitstrings from measurement
           Return best solution found
        
        Optimization Methods:
        --------------------
        
        COBYLA (Constrained Optimization BY Linear Approximations):
        - Gradient-free method (doesn't need derivatives)
        - Good for noisy cost functions (quantum measurements are inherently noisy)
        - Robust to local curvature changes
        - Slower convergence but reliable
        - Default choice for QAOA
        
        BFGS (Broyden-Fletcher-Goldfarb-Shanno):
        - Quasi-Newton method (approximates second derivatives)
        - Faster convergence when gradients available
        - Can use parameter-shift rule for quantum gradients
        - More sensitive to noise
        - Better for noiseless simulation
        
        L-BFGS-B:
        - Limited-memory BFGS with bounds
        - Memory efficient for large parameter spaces
        - Supports parameter constraints
        
        Why Classical Optimization is Necessary:
        ----------------------------------------
        The quantum circuit is parameterized by angles (γ, β). Finding the RIGHT
        angles is a classical optimization problem:
        
        minimize f(θ) where f(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
        
        - f(θ) can only be evaluated by running quantum circuit (expensive!)
        - Landscape is non-convex with many local minima
        - Quantum measurements add noise to function evaluations
        - Need smart exploration strategy to find good parameters
        
        This is why we use classical optimizers - they know how to navigate
        complex, noisy landscapes efficiently!
        
        Convergence Monitoring:
        ----------------------
        We track several metrics to monitor optimization:
        
        1. Objective Value: ⟨H⟩ at each iteration
           - Should decrease over time (we're minimizing)
           - May fluctuate due to shot noise
           - Expect logarithmic or linear convergence
        
        2. Parameter Changes: |θ_new - θ_old|
           - Large changes early (exploration)
           - Small changes later (fine-tuning)
           - Very small changes indicate convergence
        
        3. Function Tolerance: |f(θ_new) - f(θ_old)|
           - How much cost improved
           - Below threshold → declare convergence
           - Prevents unnecessary iterations
        
        Early Stopping:
        --------------
        We stop optimization early if:
        - Cost change < convergence_threshold (converged!)
        - Maximum iterations reached (prevent infinite loop)
        - Cost value plateaus for multiple iterations (stuck in local minimum)
        
        This saves computational resources and prevents overfitting to noise.
        
        Progress Tracking:
        -----------------
        During optimization, we print:
        - Iteration number
        - Current cost (expectation value)
        - Parameter values
        - Improvement from previous iteration
        
        This helps users:
        - Monitor algorithm progress
        - Debug convergence issues
        - Understand optimization behavior
        - Decide when to stop manually
        
        Algorithm Flow:
        ---------------
        
        Step 1: PROBLEM CONVERSION
            problem → QUBO matrix
            [Convert graph/constraints to mathematical formulation]
        
        Step 2: CIRCUIT PREPARATION
            QUBO → Hamiltonian → QAOA circuit
            [Build quantum circuit with parameterized gates]
        
        Step 3: PARAMETER INITIALIZATION
            θ₀ ~ random or heuristic values
            [Starting point for optimization]
        
        Step 4: OPTIMIZATION LOOP (hybrid quantum-classical)
            for iter = 1 to maxiter:
                ┌─────────────────────────────────────┐
                │ QUANTUM: Execute circuit(θ)         │  ← Quantum Device
                │          Measure ⟨H⟩                │
                └─────────────────────────────────────┘
                            ↓
                ┌─────────────────────────────────────┐
                │ CLASSICAL: Update θ to minimize ⟨H⟩ │  ← Classical CPU
                │            Check convergence        │
                └─────────────────────────────────────┘
        
        Step 5: SOLUTION SAMPLING
            Run optimized circuit many times (shots)
            Collect bitstring measurements
            Return most frequent or best bitstring
        
        Step 6: RESULT FORMATTING
            Convert bitstring to problem solution
            Calculate objective value
            Estimate energy consumption
            Return standardized result
        
        Performance Considerations:
        --------------------------
        - Each optimization iteration requires running quantum circuit
        - With shots=1024, each iteration takes ~1024 circuit executions
        - Total circuit runs = shots × maxiter (e.g., 1024 × 100 = 102,400)
        - Simulation time: seconds to minutes depending on problem size
        - Real hardware: longer due to queue times and slower execution
        
        Trade-offs:
        - More shots → better gradient estimates → slower but more accurate
        - More iterations → better optimization → longer runtime
        - More QAOA layers (p) → better approximation → deeper circuits
        
        Args:
            problem: Problem instance with to_qubo() method
                     Must implement standard problem interface
            
            p (int): Number of QAOA layers
                     More layers = more expressive but deeper circuits
                     Typical: 1-5 for NISQ devices
                     Default: 1 (simplest QAOA)
            
            maxiter (int): Maximum optimization iterations
                          More iterations = better convergence but slower
                          Typical: 50-200
                          Default: 100
            
            optimizer (str): Classical optimization method
                            Options: 'COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead'
                            Default: 'COBYLA' (gradient-free, robust to noise)
            
            initial_params (Optional[np.ndarray]): Starting parameters [γ₁, β₁, ..., γₚ, βₚ]
                                                   If None, random initialization
                                                   Can use heuristic values for better start
            
            convergence_threshold (float): Stop when |Δcost| < threshold
                                          Smaller = stricter convergence
                                          Default: 1e-6
            
            verbose (bool): Print optimization progress
                           Useful for monitoring and debugging
                           Default: True
        
        Returns:
            Dict[str, Any]: Standardized result format
                {
                    'solution': List[int],           # Best bitstring found
                    'cost': float,                   # Objective function value
                    'time_ms': int,                  # Total execution time
                    'energy_mj': float,              # Estimated energy consumption
                    'iterations': int,               # Optimization iterations
                    'metadata': {
                        'optimal_params': np.ndarray,     # Best parameters found
                        'optimization_history': List,     # Cost at each iteration
                        'convergence_reason': str,        # Why optimization stopped
                        'circuit_depth': int,             # QAOA circuit depth
                        'num_qubits': int,               # Problem size
                        'final_expectation': float,      # Final ⟨H⟩ value
                        'optimizer': str,                # Optimization method used
                        'qaoa_layers': int,              # Number of QAOA layers
                    }
                }
        
        Raises:
            QuantumSimulatorException: If QAOA execution fails
            ValueError: If problem doesn't support QUBO conversion
            ImportError: If scipy not available
        
        Example:
            >>> from src.problems.maxcut import MaxCutProblem
            >>> problem = MaxCutProblem(num_nodes=6)
            >>> problem.generate(edge_probability=0.5)
            >>> 
            >>> simulator = QuantumSimulator(shots=1024)
            >>> result = simulator.solve(problem, p=2, maxiter=50, verbose=True)
            >>> 
            >>> print(f"Best solution: {result['solution']}")
            >>> print(f"Cost: {result['cost']:.4f}")
            >>> print(f"Optimization converged in {result['iterations']} iterations")
        
        References:
            - Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
            - Zhou et al., "Quantum Approximate Optimization Algorithm: Performance,
              Mechanism, and Implementation on Near-Term Devices" (2020)
            - Scipy optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
        """
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 0: Validate inputs and check dependencies
        # ═════════════════════════════════════════════════════════════════════════
        
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "Scipy is required for classical optimization. "
                "Install with: pip install scipy"
            )
        
        # Start timing for performance tracking
        start_time = time.time()
        
        if verbose:
            print("\n" + "="*70)
            print("QAOA HYBRID QUANTUM-CLASSICAL OPTIMIZATION")
            print("="*70)
            print(f"Problem: {problem.__class__.__name__}")
            print(f"QAOA layers (p): {p}")
            print(f"Max iterations: {maxiter}")
            print(f"Optimizer: {optimizer}")
            print(f"Shots per evaluation: {self.shots}")
            print("="*70 + "\n")
        
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 1: Convert problem to QUBO formulation
        # ═════════════════════════════════════════════════════════════════════════
        
        if verbose:
            print("📊 Step 1: Converting problem to QUBO matrix...")
        
        try:
            qubo_matrix = problem.to_qubo()
            n_qubits = qubo_matrix.shape[0]
            
            if verbose:
                print(f"   ✓ QUBO matrix size: {n_qubits}×{n_qubits}")
                print(f"   ✓ Number of qubits needed: {n_qubits}")
                print(f"   ✓ Hilbert space dimension: 2^{n_qubits} = {2**n_qubits:,}")
        
        except AttributeError:
            raise ValueError(
                f"Problem {problem.__class__.__name__} does not support QUBO conversion. "
                f"Implement to_qubo() method."
            )
        except Exception as e:
            raise QuantumSimulatorException(f"Failed to convert problem to QUBO: {e}")
        
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 2: Initialize QAOA parameters
        # ═════════════════════════════════════════════════════════════════════════
        
        if verbose:
            print(f"\n🎲 Step 2: Initializing QAOA parameters...")
        
        n_params = 2 * p  # Each layer has γ (cost) and β (mixer) parameters
        
        if initial_params is not None:
            if len(initial_params) != n_params:
                raise ValueError(
                    f"initial_params has {len(initial_params)} elements, "
                    f"expected {n_params} for p={p} layers"
                )
            params = initial_params.copy()
            if verbose:
                print(f"   ✓ Using provided initial parameters: {params}")
        else:
            # Random initialization with heuristic bounds
            # γ (cost angles): typically in [0, π]
            # β (mixer angles): typically in [0, π/2]
            params = np.zeros(n_params)
            for i in range(p):
                params[2*i] = np.random.uniform(0, np.pi)        # γ_i
                params[2*i + 1] = np.random.uniform(0, np.pi/2)  # β_i
            
            if verbose:
                print(f"   ✓ Random initialization: {params}")
                print(f"   ✓ Parameter count: {n_params} (γ₁, β₁, ..., γ_{p}, β_{p})")
        
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 3: Build QAOA circuit
        # ═════════════════════════════════════════════════════════════════════════
        
        if verbose:
            print(f"\n⚛️  Step 3: Building QAOA quantum circuit...")
        
        try:
            circuit = self._build_qaoa_circuit(qubo_matrix, params, p=p)
            
            if verbose:
                print(f"   ✓ Circuit built successfully")
                print(f"   ✓ Backend: {self.backend}")
                print(f"   ✓ Estimated circuit depth: ~{p * (n_qubits + len(qubo_matrix.flatten()))}")
        
        except Exception as e:
            raise QuantumSimulatorException(f"Failed to build QAOA circuit: {e}")
        
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 4: Set up optimization tracking
        # ═════════════════════════════════════════════════════════════════════════
        
        optimization_history = []  # Track cost at each iteration
        iteration_count = [0]      # Mutable counter for callback
        best_cost = [float('inf')] # Track best cost found
        best_params = [params.copy()]  # Track best parameters
        prev_cost = [None]         # For convergence detection
        
        # Callback function to track optimization progress
        def callback(xk):
            """
            Called after each optimization iteration.
            
            This function is invoked by the scipy optimizer after each step,
            allowing us to monitor progress, check convergence, and provide
            user feedback.
            
            Args:
                xk: Current parameter values
            """
            iteration_count[0] += 1
            
            # Evaluate cost at current parameters
            # Note: We re-evaluate because scipy doesn't always provide cost in callback
            current_cost = self._measure_qaoa_expectation(circuit, xk)
            optimization_history.append(current_cost)
            
            # Track best solution found so far
            if current_cost < best_cost[0]:
                best_cost[0] = current_cost
                best_params[0] = xk.copy()
            
            # Calculate improvement from previous iteration
            if prev_cost[0] is not None:
                improvement = prev_cost[0] - current_cost
                improvement_pct = (improvement / abs(prev_cost[0])) * 100 if prev_cost[0] != 0 else 0
            else:
                improvement = 0
                improvement_pct = 0
            
            prev_cost[0] = current_cost
            
            # Print progress if verbose
            if verbose:
                print(f"   Iter {iteration_count[0]:3d} | "
                      f"Cost: {current_cost:+.6f} | "
                      f"Improvement: {improvement:+.6f} ({improvement_pct:+.2f}%) | "
                      f"Best: {best_cost[0]:+.6f}")
            
            # Check for early stopping (convergence)
            if iteration_count[0] > 1 and abs(improvement) < convergence_threshold:
                if verbose:
                    print(f"\n   🎯 Early stopping: Converged! (improvement < {convergence_threshold})")
                return True  # Signal to stop optimization
        
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 5: Run hybrid quantum-classical optimization loop
        # ═════════════════════════════════════════════════════════════════════════
        
        if verbose:
            print(f"\n🔄 Step 4: Running hybrid optimization loop...")
            print(f"   This is where quantum and classical computing work together!")
            print(f"   Quantum: Evaluates cost function via circuit execution")
            print(f"   Classical: Adjusts parameters to minimize cost\n")
        
        try:
            # Define objective function for classical optimizer
            def objective(params_opt):
                """
                Objective function for classical optimization.
                
                This function is called by the scipy optimizer. It:
                1. Takes parameter values from classical optimizer
                2. Passes them to quantum circuit
                3. Executes quantum circuit and measures expectation
                4. Returns cost value to classical optimizer
                
                This is the quantum-classical interface!
                """
                return self._measure_qaoa_expectation(circuit, params_opt)
            
            # Run classical optimization
            # This is where the magic happens - the optimizer will repeatedly:
            # 1. Propose new parameters
            # 2. Call objective() which runs quantum circuit
            # 3. Receive cost value
            # 4. Adjust parameters based on cost
            # 5. Repeat until convergence or max iterations
            
            result = minimize(
                objective,              # Function to minimize (runs quantum circuit)
                params,                 # Initial parameters
                method=optimizer,       # Optimization algorithm
                callback=callback,      # Track progress
                options={
                    'maxiter': maxiter,    # Maximum iterations
                    'disp': False,         # Don't show scipy's own messages
                }
            )
            
            # Extract optimization results
            optimal_params = best_params[0]  # Use best params found (not final)
            final_cost = best_cost[0]
            converged = result.success
            convergence_reason = result.message if hasattr(result, 'message') else 'Unknown'
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Optimization {'CONVERGED' if converged else 'COMPLETED'}")
                print(f"{'='*70}")
                print(f"Total iterations: {iteration_count[0]}")
                print(f"Final cost: {final_cost:.6f}")
                print(f"Initial cost: {optimization_history[0]:.6f}")
                print(f"Total improvement: {optimization_history[0] - final_cost:.6f}")
                print(f"Convergence reason: {convergence_reason}")
                print(f"{'='*70}\n")
        
        except Exception as e:
            raise QuantumSimulatorException(f"Optimization failed: {e}")
        
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 6: Sample solutions using optimal parameters
        # ═════════════════════════════════════════════════════════════════════════
        
        if verbose:
            print(f"🎯 Step 5: Sampling solutions with optimal parameters...")
        
        # Build sampling circuit (without expectation value, just measurements)
        device = self._create_device(n_qubits)
        
        @qml.qnode(device)
        def sampling_circuit(params_sample):
            """Circuit for sampling final solutions (returns samples, not expectation)."""
            # Build the same QAOA circuit
            hamiltonian = self._qubo_to_hamiltonian(qubo_matrix)
            h_coeffs, h_ops = hamiltonian.terms()
            
            # Initialize in superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply QAOA layers
            for layer in range(p):
                gamma = params_sample[2 * layer]
                beta = params_sample[2 * layer + 1]
                
                # Cost layer
                for coeff, op in zip(h_coeffs, h_ops):
                    qubits = op.wires.tolist()
                    if len(qubits) == 1:
                        qml.RZ(2 * gamma * coeff, wires=qubits[0])
                    elif len(qubits) == 2:
                        q_i, q_j = qubits
                        angle = 2 * gamma * coeff
                        qml.CNOT(wires=[q_i, q_j])
                        qml.RZ(angle, wires=q_j)
                        qml.CNOT(wires=[q_i, q_j])
                
                # Mixer layer
                for i in range(n_qubits):
                    qml.RX(2 * beta, wires=i)
            
            # Measure all qubits
            return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Sample multiple times
        samples = sampling_circuit(optimal_params)
        
        # Convert samples to bitstrings
        # Pennylane samples give ±1, convert to 0/1
        # -1 (eigenvalue of Z for |1⟩) → 1 (binary)
        # +1 (eigenvalue of Z for |0⟩) → 0 (binary)
        if isinstance(samples, list):
            # Multiple measurements
            bitstrings = []
            for _ in range(self.shots):
                sample = sampling_circuit(optimal_params)
                bitstring = [(1 - int(s)) // 2 for s in sample]
                bitstrings.append(bitstring)
        else:
            # Single shot (convert single sample)
            bitstrings = [[(1 - int(s)) // 2 for s in samples]]
        
        # Find most common bitstring (or evaluate all and pick best)
        best_solution = None
        best_solution_cost = float('inf')
        
        for bitstring in bitstrings:
            # Evaluate this solution's cost using QUBO
            cost = 0
            for i in range(n_qubits):
                for j in range(n_qubits):
                    cost += qubo_matrix[i, j] * bitstring[i] * bitstring[j]
            
            if cost < best_solution_cost:
                best_solution_cost = cost
                best_solution = bitstring
        
        if verbose:
            print(f"   ✓ Sampled {len(bitstrings)} solutions")
            print(f"   ✓ Best solution found: {best_solution}")
            print(f"   ✓ Best solution cost: {best_solution_cost:.6f}")
        
        # ═════════════════════════════════════════════════════════════════════════
        # STEP 7: Calculate execution metrics and format results
        # ═════════════════════════════════════════════════════════════════════════
        
        end_time = time.time()
        execution_time_ms = int((end_time - start_time) * 1000)
        
        # Estimate energy consumption
        # Rough estimates for quantum simulation:
        # - Classical CPU simulation: ~10-50 mJ per circuit execution
        # - Depends on qubit count, circuit depth, shots
        total_circuit_executions = iteration_count[0] * self.shots + self.shots  # optimization + sampling
        energy_per_circuit_mj = 0.02 * (2 ** min(n_qubits, 10))  # Exponential scaling (capped)
        estimated_energy_mj = total_circuit_executions * energy_per_circuit_mj
        
        # Prepare metadata
        metadata = {
            'optimal_params': optimal_params.tolist(),
            'optimization_history': optimization_history,
            'convergence_reason': convergence_reason,
            'converged': converged,
            'circuit_depth': p * (n_qubits + len(h_coeffs) * 2),  # Approximate
            'num_qubits': n_qubits,
            'final_expectation': final_cost,
            'optimizer': optimizer,
            'qaoa_layers': p,
            'total_iterations': iteration_count[0],
            'total_circuit_executions': total_circuit_executions,
            'backend': self.backend,
            'shots': self.shots,
            'initial_cost': optimization_history[0] if optimization_history else None,
            'improvement': (optimization_history[0] - final_cost) if optimization_history else 0,
        }
        
        # Format standardized result
        result_dict = {
            'solution': best_solution,
            'cost': float(best_solution_cost),
            'time_ms': execution_time_ms,
            'energy_mj': estimated_energy_mj,
            'iterations': iteration_count[0],
            'metadata': metadata
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FINAL RESULTS")
            print(f"{'='*70}")
            print(f"Solution: {best_solution}")
            print(f"Cost: {best_solution_cost:.6f}")
            print(f"Execution time: {execution_time_ms} ms")
            print(f"Energy consumption: {estimated_energy_mj:.2f} mJ")
            print(f"Total circuit executions: {total_circuit_executions:,}")
            print(f"{'='*70}\n")
        
        return result_dict
    
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get information about this quantum simulator.
        
        Returns comprehensive metadata about the solver's capabilities,
        configuration, and supported problem types.
        
        Returns:
            Dict containing:
                - solver_type: 'quantum'
                - solver_name: 'quantum_simulator_pennylane'
                - version: Pennylane version
                - backend: Device backend name
                - capabilities: Supported operations
                - noise_model: Current noise configuration
                - supported_problems: List of problem types
                - limitations: Known constraints
        """
        info = {
            'solver_type': 'quantum',
            'solver_name': 'quantum_simulator_pennylane',
            'backend': self.backend,
            'shots': self.shots,
            'max_circuit_depth': self.max_circuit_depth,
            'noise_model': self.noise_model.copy(),
            'supported_problems': [
                'maxcut',
                'graph_coloring',
                'vertex_cover',
                'sat',
                'tsp',
                'portfolio'
            ],
            'algorithms': [
                'qaoa',  # Quantum Approximate Optimization Algorithm
                'vqe',   # Variational Quantum Eigensolver
            ],
            'capabilities': {
                'quantum_gates': ['RX', 'RY', 'RZ', 'CNOT', 'H', 'X', 'Y', 'Z'],
                'measurement': 'computational_basis',
                'noise_simulation': self.backend == 'default.mixed',
                'photonic_modeling': True,
            },
            'resource_requirements': {
                'qubits': 'problem-dependent (typically 10-100)',
                'circuit_depth': f'up to {self.max_circuit_depth}',
                'memory_mb': 'exponential in qubit count: ~2^n bytes',
                'time_per_shot_ms': 0.1,  # Rough estimate
            },
            'limitations': [
                f'Maximum circuit depth: {self.max_circuit_depth}',
                f'Photon loss rate: {self.noise_model["photon_loss_rate"]*100:.2f}%',
                'Classical simulation (not real quantum hardware)',
                'Exponential memory scaling with qubits',
            ]
        }
        
        # Add Pennylane version if available
        try:
            info['pennylane_version'] = qml.__version__
        except:
            info['pennylane_version'] = 'unknown'
        
        return info


# Module-level convenience functions

def create_simulator(
    backend: str = 'default.qubit',
    shots: int = 1024,
    noise_model: Optional[Dict[str, float]] = None
) -> QuantumSimulator:
    """
    Convenience function to create a quantum simulator instance.
    
    Args:
        backend: Pennylane backend name
        shots: Number of circuit executions
        noise_model: Optional custom noise parameters
        
    Returns:
        QuantumSimulator instance
        
    Example:
        >>> sim = create_simulator(shots=2048)
        >>> result = sim.solve(problem)
    """
    return QuantumSimulator(
        backend=backend,
        shots=shots,
        noise_model=noise_model
    )


def get_photonic_noise_profile(quality: str = 'standard') -> Dict[str, float]:
    """
    Get predefined photonic noise profiles for different quality levels.
    
    Args:
        quality: One of 'ideal', 'standard', 'realistic', 'noisy'
        
    Returns:
        Dict of noise parameters
        
    Profiles:
        - 'ideal': Near-perfect gates (research goal)
        - 'standard': Current state-of-art (default)
        - 'realistic': Expected near-term systems
        - 'noisy': Conservative estimates
    """
    profiles = {
        'ideal': {
            'photon_loss_rate': 0.0001,
            'detection_efficiency': 0.999,
            'single_qubit_gate_fidelity': 0.9999,
            'two_qubit_gate_fidelity': 0.999,
            'measurement_fidelity': 0.999,
            'thermal_photon_rate': 0.0001,
        },
        'standard': QuantumSimulator.DEFAULT_PHOTONIC_NOISE,
        'realistic': {
            'photon_loss_rate': 0.01,
            'detection_efficiency': 0.95,
            'single_qubit_gate_fidelity': 0.998,
            'two_qubit_gate_fidelity': 0.93,
            'measurement_fidelity': 0.96,
            'thermal_photon_rate': 0.002,
        },
        'noisy': {
            'photon_loss_rate': 0.02,
            'detection_efficiency': 0.90,
            'single_qubit_gate_fidelity': 0.99,
            'two_qubit_gate_fidelity': 0.85,
            'measurement_fidelity': 0.93,
            'thermal_photon_rate': 0.005,
        }
    }
    
    if quality not in profiles:
        raise ValueError(
            f"Unknown quality level: {quality}. "
            f"Choose from: {list(profiles.keys())}"
        )
    
    return profiles[quality].copy()
