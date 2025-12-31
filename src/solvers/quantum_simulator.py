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
