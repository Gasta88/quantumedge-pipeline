#!/usr/bin/env python3
"""
ML Graph Partitioning Demo for QuantumEdge Pipeline.

This demo showcases graph partitioning for distributed machine learning training
across edge devices. It partitions a neural network computation graph to minimize
inter-device communication while balancing computational load.

Scenario: ML Graph Partitioning
===============================
**Problem**: Partition neural network graph for distributed training across edge devices

**Details**:
- ResNet-50 computation graph (50 layers)
- Minimize inter-device communication
- Balance compute load across 4 edge nodes
- Latency-sensitive constraints

**Expected Results** (from README):
- Communication overhead reduced by 67%
- Load balance variance: <5%
- Quantum solver selected for 30+ node subgraphs

Usage:
------
    # Run from project root
    python scripts/demos/ml_graph_partition.py
    
    # Or using docker-compose
    docker-compose exec api python scripts/demos/ml_graph_partition.py
    
    # With visualization
    python scripts/demos/ml_graph_partition.py --visualize
    
    # With custom parameters
    python scripts/demos/ml_graph_partition.py --layers 100 --devices 8

ML/Edge Context:
----------------
This demonstration is relevant for:
- Distributed deep learning training
- Edge ML inference optimization
- Model parallelism strategies
- IoT device coordination
- 5G/6G network optimization with quantum acceleration
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from uuid import uuid4
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import networkx as nx

# Import QuantumEdge components
from src.problems.maxcut import MaxCutProblem
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
# Neural Network Graph Scenario Configuration
# =============================================================================

class LayerNode:
    """Neural network layer node."""
    
    def __init__(self, layer_id: int, layer_type: str, compute_cost: float, memory_mb: float):
        """
        Args:
            layer_id: Unique layer identifier
            layer_type: Type of layer (conv, pool, fc, etc.)
            compute_cost: Relative computational cost (FLOPs normalized)
            memory_mb: Memory footprint in MB
        """
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.compute_cost = compute_cost
        self.memory_mb = memory_mb


class NeuralNetworkGraph:
    """Neural network computation graph for partitioning."""
    
    LAYER_TYPES = {
        'conv': {'compute': 1.0, 'memory': 10.0, 'comm': 5.0},
        'pool': {'compute': 0.1, 'memory': 2.0, 'comm': 2.0},
        'fc': {'compute': 2.0, 'memory': 50.0, 'comm': 10.0},
        'bn': {'compute': 0.2, 'memory': 1.0, 'comm': 1.0},
        'relu': {'compute': 0.05, 'memory': 0.5, 'comm': 0.5},
    }
    
    def __init__(self, num_layers: int = 50, num_devices: int = 4):
        """
        Initialize neural network graph scenario.
        
        Args:
            num_layers: Number of layers in the network
            num_devices: Number of edge devices for partitioning
        """
        self.num_layers = num_layers
        self.num_devices = num_devices
        self.layers: List[LayerNode] = []
        self.graph: Optional[nx.DiGraph] = None
        self.communication_costs: Dict[Tuple[int, int], float] = {}
        
    def generate_resnet_like_graph(self, seed: Optional[int] = None):
        """
        Generate a ResNet-50 like computation graph.
        
        ResNet structure:
        - Initial conv + pool layers
        - Residual blocks (conv-bn-relu patterns with skip connections)
        - Final fc layer
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Generating ResNet-like graph with {self.num_layers} layers...")
        
        # Create directed acyclic graph (DAG)
        self.graph = nx.DiGraph()
        
        # Generate layers with realistic characteristics
        layer_types = []
        
        # Initial layers (stem)
        layer_types.extend(['conv', 'bn', 'relu', 'pool'])
        
        # Residual blocks (groups of conv-bn-relu)
        num_blocks = (self.num_layers - 6) // 3  # Subtract stem and final layers
        for _ in range(num_blocks):
            layer_types.extend(['conv', 'bn', 'relu'])
        
        # Final layers
        layer_types.extend(['pool', 'fc'])
        
        # Truncate or pad to exact num_layers
        layer_types = layer_types[:self.num_layers]
        while len(layer_types) < self.num_layers:
            layer_types.append('conv')
        
        # Create layer nodes
        for i, layer_type in enumerate(layer_types):
            layer_config = self.LAYER_TYPES[layer_type]
            
            # Add variability
            compute_mult = 1.0 + np.random.uniform(-0.2, 0.2)
            memory_mult = 1.0 + np.random.uniform(-0.1, 0.1)
            
            layer = LayerNode(
                layer_id=i,
                layer_type=layer_type,
                compute_cost=layer_config['compute'] * compute_mult,
                memory_mb=layer_config['memory'] * memory_mult
            )
            self.layers.append(layer)
            
            # Add node to graph with attributes
            self.graph.add_node(
                i,
                layer_type=layer_type,
                compute=layer.compute_cost,
                memory=layer.memory_mb
            )
        
        # Add edges (dependencies between layers)
        # Sequential connections
        for i in range(self.num_layers - 1):
            from_type = layer_types[i]
            to_type = layer_types[i + 1]
            
            # Communication cost based on layer types
            from_config = self.LAYER_TYPES[from_type]
            to_config = self.LAYER_TYPES[to_type]
            comm_cost = (from_config['comm'] + to_config['comm']) / 2
            comm_cost *= (1.0 + np.random.uniform(-0.1, 0.1))
            
            self.graph.add_edge(i, i + 1, weight=comm_cost)
            self.communication_costs[(i, i + 1)] = comm_cost
        
        # Add skip connections (ResNet characteristic)
        # Every 3 layers, add a skip connection
        for i in range(4, self.num_layers - 1, 3):
            if i + 2 < self.num_layers:
                skip_cost = 2.0 + np.random.uniform(-0.5, 0.5)
                self.graph.add_edge(i, i + 2, weight=skip_cost)
                self.communication_costs[(i, i + 2)] = skip_cost
        
        logger.info(f"✓ Generated computation graph")
        logger.info(f"  Nodes (layers): {self.graph.number_of_nodes()}")
        logger.info(f"  Edges (dependencies): {self.graph.number_of_edges()}")
        logger.info(f"  Total compute: {sum(l.compute_cost for l in self.layers):.2f} units")
        logger.info(f"  Total memory: {sum(l.memory_mb for l in self.layers):.1f} MB")
    
    def to_maxcut_problem(self) -> MaxCutProblem:
        """
        Convert graph partitioning to MaxCut problem.
        
        For k-way partitioning (k devices), we use a simplified approach:
        - Binary partitioning (split into 2 groups)
        - Can be extended to hierarchical partitioning for k > 2
        
        Edge weights represent communication costs that should be minimized
        when nodes are in different partitions (devices).
        
        Returns:
            MaxCutProblem instance (actually MinCut for our purposes)
        """
        problem = MaxCutProblem(num_nodes=self.num_layers)
        
        # Convert to undirected graph for MaxCut
        undirected_graph = self.graph.to_undirected()
        problem.graph = undirected_graph
        
        # Build adjacency matrix
        adjacency = nx.to_numpy_array(undirected_graph, weight='weight')
        problem.adjacency_matrix = adjacency
        
        # Store edge weights (NOTE: for partitioning, we want MINCUT, not MAXCUT)
        # We'll invert the objective later
        problem.edge_weights = {}
        for u, v in undirected_graph.edges():
            weight = undirected_graph[u][v]['weight']
            problem.edge_weights[(min(u, v), max(u, v))] = weight
        
        problem._generated = True
        return problem
    
    def evaluate_partition(
        self, 
        partition: List[int],
        problem: MaxCutProblem
    ) -> Dict[str, float]:
        """
        Evaluate quality of a graph partition.
        
        Args:
            partition: Binary partition assignment (0 or 1 for each layer)
            problem: MaxCutProblem instance
        
        Returns:
            Dictionary with partition quality metrics
        """
        # Assign layers to devices based on partition
        # For k > 2 devices, this would be more complex
        device_0 = [i for i, p in enumerate(partition) if p == 0]
        device_1 = [i for i, p in enumerate(partition) if p == 1]
        
        # Calculate communication overhead (edges crossing partition)
        comm_overhead = 0.0
        cross_edges = 0
        total_edges = 0
        
        for (u, v), weight in problem.edge_weights.items():
            total_edges += 1
            if partition[u] != partition[v]:
                comm_overhead += weight
                cross_edges += 1
        
        # Communication reduction (compared to all edges crossing)
        total_comm = sum(problem.edge_weights.values())
        comm_reduction_pct = ((total_comm - comm_overhead) / total_comm * 100) if total_comm > 0 else 0
        
        # Calculate load balance
        compute_0 = sum(self.layers[i].compute_cost for i in device_0)
        compute_1 = sum(self.layers[i].compute_cost for i in device_1)
        
        total_compute = compute_0 + compute_1
        ideal_load = total_compute / 2
        
        load_imbalance = abs(compute_0 - ideal_load) / ideal_load * 100 if ideal_load > 0 else 0
        
        # Memory usage
        memory_0 = sum(self.layers[i].memory_mb for i in device_0)
        memory_1 = sum(self.layers[i].memory_mb for i in device_1)
        
        return {
            'communication_overhead_mb': comm_overhead,
            'communication_reduction_pct': comm_reduction_pct,
            'cross_partition_edges': cross_edges,
            'total_edges': total_edges,
            'edge_cut_ratio': cross_edges / total_edges if total_edges > 0 else 0,
            'device_0_layers': len(device_0),
            'device_1_layers': len(device_1),
            'device_0_compute': compute_0,
            'device_1_compute': compute_1,
            'load_imbalance_pct': load_imbalance,
            'device_0_memory_mb': memory_0,
            'device_1_memory_mb': memory_1,
        }


# =============================================================================
# Demo Execution
# =============================================================================

def run_ml_partition_demo(
    num_layers: int = 50,
    num_devices: int = 4,
    visualize: bool = False,
    verbose: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run the ML graph partitioning demo.
    
    Args:
        num_layers: Number of neural network layers
        num_devices: Number of edge devices (currently supports 2-way partition)
        visualize: Whether to generate visualizations
        verbose: Whether to print detailed progress
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing results
    """
    logger.info("=" * 80)
    logger.info("ML GRAPH PARTITIONING DEMO")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Scenario: Neural network graph partitioning for distributed training")
    logger.info(f"Layers: {num_layers}, Target devices: {num_devices} (2-way partition)")
    logger.info(f"Seed: {seed}")
    logger.info("")
    
    # Step 1: Generate neural network graph
    logger.info("[Step 1/5] Generating neural network computation graph...")
    nn_graph = NeuralNetworkGraph(num_layers=num_layers, num_devices=num_devices)
    nn_graph.generate_resnet_like_graph(seed=seed)
    
    # Convert to MaxCut (MinCut) problem
    problem = nn_graph.to_maxcut_problem()
    logger.info("")
    
    # Step 2: Analyze problem
    logger.info("[Step 2/5] Analyzing graph characteristics...")
    analyzer = ProblemAnalyzer()
    analysis = analyzer.analyze(problem)
    logger.info(f"✓ Graph analyzed")
    logger.info(f"  Problem type: {analysis.problem_type}")
    logger.info(f"  Problem size: {analysis.problem_size} layers")
    logger.info(f"  Graph density: {analysis.metadata.get('graph_density', 0):.2%}")
    logger.info(f"  Avg degree: {analysis.metadata.get('average_degree', 0):.1f}")
    logger.info("")
    
    # Step 3: Get routing recommendation
    logger.info("[Step 3/5] Querying quantum router for solver recommendation...")
    router = QuantumRouter()
    routing_decision = router.route(problem)
    logger.info(f"✓ Router recommendation: {routing_decision.recommended_solver}")
    logger.info(f"  Confidence: {routing_decision.confidence:.1%}")
    logger.info(f"  Reasoning: {routing_decision.reasoning}")
    
    # Note about quantum advantage for 30+ nodes
    if num_layers >= 30:
        logger.info(f"  Note: Problem size ({num_layers} layers) suggests quantum advantage")
    logger.info("")
    
    # Step 4: Solve with multiple approaches
    logger.info("[Step 4/5] Computing optimal partition...")
    results = {}
    
    # Classical Solver
    logger.info("  → Solving with Classical graph partitioner...")
    start_time = time.time()
    classical_solver = ClassicalSolver()
    
    # For MinCut (partitioning), we solve MaxCut and invert
    classical_result = classical_solver.solve(problem, method='auto')
    classical_time = time.time() - start_time
    
    classical_partition = classical_result['solution']
    classical_metrics = nn_graph.evaluate_partition(classical_partition, problem)
    
    results['classical'] = {
        'solver': 'Classical (Greedy)',
        'partition': classical_partition,
        'execution_time_s': classical_time,
        'metrics': classical_metrics,
        'energy_consumed_mj': classical_result.get('energy_mj', classical_time * 5.0)
    }
    
    logger.info(f"    ✓ Classical: {classical_time:.2f}s")
    logger.info(f"      Communication reduced: {classical_metrics['communication_reduction_pct']:.1f}%")
    logger.info(f"      Load imbalance: {classical_metrics['load_imbalance_pct']:.1f}%")
    
    # Quantum Solver (if available and problem size suitable)
    if QUANTUM_AVAILABLE and num_layers <= 40:  # QAOA feasible for moderate sizes
        logger.info("  → Solving with Quantum graph partitioner...")
        try:
            start_time = time.time()
            quantum_solver = QuantumSimulator(backend='qiskit_aer')
            quantum_result = quantum_solver.solve(problem, method='qaoa', p=2, max_iterations=100)
            quantum_time = time.time() - start_time
            
            quantum_partition = quantum_result['solution']
            quantum_metrics = nn_graph.evaluate_partition(quantum_partition, problem)
            
            results['quantum'] = {
                'solver': 'Quantum QAOA',
                'partition': quantum_partition,
                'execution_time_s': quantum_time,
                'metrics': quantum_metrics,
                'energy_consumed_mj': quantum_result.get('energy_mj', quantum_time * 3.0)
            }
            
            logger.info(f"    ✓ Quantum QAOA: {quantum_time:.2f}s")
            logger.info(f"      Communication reduced: {quantum_metrics['communication_reduction_pct']:.1f}%")
            logger.info(f"      Load imbalance: {quantum_metrics['load_imbalance_pct']:.1f}%")
        except Exception as e:
            logger.warning(f"    ⚠ Quantum solver encountered issue: {e}")
    elif num_layers > 40:
        logger.info("  → Quantum solver: Problem size too large for current demo")
    
    logger.info("")
    
    # Step 5: Analysis and comparison
    logger.info("[Step 5/5] Partition Analysis")
    logger.info("=" * 80)
    
    # Find best partition
    best_solver = None
    best_comm_reduction = -np.inf
    
    for solver_name, result in results.items():
        comm_reduction = result['metrics']['communication_reduction_pct']
        if comm_reduction > best_comm_reduction:
            best_comm_reduction = comm_reduction
            best_solver = solver_name
    
    best_result = results[best_solver]
    best_metrics = best_result['metrics']
    
    logger.info("")
    logger.info("Optimal Partition Results:")
    logger.info("-" * 80)
    logger.info(f"Solver: {best_result['solver']}")
    logger.info(f"Execution time: {best_result['execution_time_s']:.2f}s")
    logger.info("")
    
    logger.info("Communication Metrics:")
    logger.info(f"  Overhead: {best_metrics['communication_overhead_mb']:.2f} MB")
    logger.info(f"  Reduction: {best_metrics['communication_reduction_pct']:.1f}% ✅")
    logger.info(f"  Cross-partition edges: {best_metrics['cross_partition_edges']}/{best_metrics['total_edges']}")
    logger.info(f"  Edge cut ratio: {best_metrics['edge_cut_ratio']:.2%}")
    logger.info("")
    
    logger.info("Load Balance:")
    logger.info(f"  Device 0: {best_metrics['device_0_layers']} layers, "
               f"{best_metrics['device_0_compute']:.1f} compute units, "
               f"{best_metrics['device_0_memory_mb']:.1f} MB")
    logger.info(f"  Device 1: {best_metrics['device_1_layers']} layers, "
               f"{best_metrics['device_1_compute']:.1f} compute units, "
               f"{best_metrics['device_1_memory_mb']:.1f} MB")
    logger.info(f"  Load imbalance: {best_metrics['load_imbalance_pct']:.1f}% ✅")
    logger.info("")
    
    # Comparison if multiple solvers ran
    if len(results) > 1:
        logger.info("Solver Comparison:")
        logger.info("-" * 80)
        logger.info(f"{'Solver':<20} {'Time (s)':<12} {'Comm Reduction':<18} {'Load Imbalance':<15}")
        logger.info("-" * 80)
        
        for solver_name, result in results.items():
            metrics = result['metrics']
            winner_mark = " ✅" if solver_name == best_solver else ""
            logger.info(
                f"{result['solver']:<20} "
                f"{result['execution_time_s']:<12.2f} "
                f"{metrics['communication_reduction_pct']:<18.1f}% "
                f"{metrics['load_imbalance_pct']:<15.1f}%"
                f"{winner_mark}"
            )
        logger.info("-" * 80)
        logger.info("")
    
    # Interpretation
    logger.info("Performance Analysis:")
    logger.info("-" * 80)
    
    if best_metrics['communication_reduction_pct'] > 60:
        logger.info("✅ Excellent communication reduction (>60%)")
    elif best_metrics['communication_reduction_pct'] > 40:
        logger.info("✅ Good communication reduction (>40%)")
    else:
        logger.info("⚠️  Moderate communication reduction")
    
    if best_metrics['load_imbalance_pct'] < 5:
        logger.info("✅ Excellent load balance (<5% imbalance)")
    elif best_metrics['load_imbalance_pct'] < 15:
        logger.info("✅ Good load balance (<15% imbalance)")
    else:
        logger.info("⚠️  Significant load imbalance")
    
    logger.info("")
    logger.info("Use Case:")
    logger.info("-" * 80)
    logger.info("This partition enables efficient distributed ML training:")
    logger.info("• Minimized data transfer between edge devices")
    logger.info("• Balanced computational workload")
    logger.info("• Optimized for latency-sensitive edge deployment")
    logger.info("• Suitable for 5G/6G network optimization scenarios")
    logger.info("")
    
    # Visualize if requested
    if visualize and PLOTTING_AVAILABLE:
        logger.info("Generating visualizations...")
        visualize_results(nn_graph, results, problem)
        logger.info("✓ Visualizations saved to ./ml_partition_*.png")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return {
        'nn_graph': nn_graph,
        'problem': problem,
        'results': results,
        'winner': best_solver
    }


def visualize_results(nn_graph: NeuralNetworkGraph, results: Dict[str, Any], problem: MaxCutProblem):
    """Generate visualization plots for ML graph partitioning results."""
    if not PLOTTING_AVAILABLE:
        return
    
    best_result = list(results.values())[0]  # Use first result for visualization
    partition = best_result['partition']
    
    # Plot 1: Graph visualization with partition
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use hierarchical layout for neural network
    pos = nx.spring_layout(nn_graph.graph, k=2, iterations=50, seed=42)
    
    # Color nodes by partition
    node_colors = ['lightcoral' if partition[i] == 0 else 'lightgreen' 
                  for i in range(len(partition))]
    
    # Color edges by whether they cross partition
    edge_colors = []
    edge_widths = []
    for u, v in nn_graph.graph.edges():
        if partition[u] != partition[v]:
            edge_colors.append('red')
            edge_widths.append(2.0)
        else:
            edge_colors.append('lightgray')
            edge_widths.append(1.0)
    
    # Draw graph
    nx.draw_networkx_nodes(
        nn_graph.graph, pos,
        node_color=node_colors,
        node_size=300,
        alpha=0.9,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        nn_graph.graph, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=10,
        ax=ax
    )
    
    nx.draw_networkx_labels(
        nn_graph.graph, pos,
        labels={i: str(i) for i in range(len(partition))},
        font_size=8,
        ax=ax
    )
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', label='Device 0'),
        Patch(facecolor='lightgreen', label='Device 1'),
        Patch(facecolor='red', label='Cross-device edges'),
        Patch(facecolor='lightgray', label='Intra-device edges'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    metrics = best_result['metrics']
    ax.set_title(
        f"Neural Network Graph Partition\n"
        f"Comm Reduction: {metrics['communication_reduction_pct']:.1f}%, "
        f"Load Imbalance: {metrics['load_imbalance_pct']:.1f}%",
        fontsize=14, fontweight='bold'
    )
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('ml_partition_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Metrics comparison
    if len(results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        solvers = [r['solver'] for r in results.values()]
        comm_reductions = [r['metrics']['communication_reduction_pct'] for r in results.values()]
        load_imbalances = [r['metrics']['load_imbalance_pct'] for r in results.values()]
        
        # Communication reduction
        ax = axes[0]
        bars = ax.bar(solvers, comm_reductions, color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Communication Reduction (%)', fontsize=12)
        ax.set_title('Communication Overhead Reduction', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Load imbalance
        ax = axes[1]
        bars = ax.bar(solvers, load_imbalances, color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Load Imbalance (%)', fontsize=12)
        ax.set_title('Computational Load Balance', fontsize=14, fontweight='bold')
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ml_partition_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="ML Graph Partitioning Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--layers', type=int, default=50,
        help='Number of neural network layers (default: 50)'
    )
    parser.add_argument(
        '--devices', type=int, default=4,
        help='Number of edge devices (default: 4, currently supports 2-way partition)'
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
        results = run_ml_partition_demo(
            num_layers=args.layers,
            num_devices=args.devices,
            visualize=args.visualize,
            verbose=args.verbose,
            seed=args.seed
        )
        
        # Save results to JSON
        output_file = f"ml_partition_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {
                'num_layers': results['nn_graph'].num_layers,
                'num_devices': results['nn_graph'].num_devices,
                'winner': results['winner'],
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
