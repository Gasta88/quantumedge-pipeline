"""
Benchmarking Script for QuantumEdge Pipeline Solvers.

This script performs comprehensive performance benchmarking of classical and quantum
solvers across various problem sizes, providing comparative analysis of execution time,
energy consumption, and solution quality.

Purpose:
--------
- Compare classical solvers (greedy, simulated annealing) vs quantum solvers (QAOA)
- Identify problem sizes where quantum advantage emerges
- Generate performance visualizations and analysis reports
- Store results in database for long-term tracking
- Export results to CSV for external analysis

Key Metrics:
------------
1. Execution Time (ms): Total solver runtime
2. Energy Consumption (mJ): Estimated energy usage
3. Solution Quality (0-1): Optimality score
4. Quantum Advantage Ratio: Time_classical / Time_quantum

Usage:
------
    # Run benchmark with default problem sizes
    python scripts/benchmark_solvers.py
    
    # Run benchmark with custom sizes
    python scripts/benchmark_solvers.py --sizes 10 20 30 40 50
    
    # Run benchmark and generate plots
    python scripts/benchmark_solvers.py --plot
    
    # Run benchmark with custom repetitions
    python scripts/benchmark_solvers.py --repetitions 5

Example:
--------
    >>> from scripts.benchmark_solvers import run_benchmark, analyze_results
    >>> 
    >>> # Run benchmark
    >>> results = run_benchmark(problem_sizes=[10, 20, 30])
    >>> 
    >>> # Analyze results
    >>> analysis = analyze_results()
    >>> print(analysis['quantum_advantage_sizes'])
    >>> 
    >>> # Generate plots
    >>> plot_performance()
    >>> 
    >>> # Generate report
    >>> generate_report()
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import QuantumEdge components
from src.problems.maxcut import MaxCutProblem
from src.solvers.classical_solver import ClassicalSolver
from src.solvers.quantum_simulator import QuantumSimulator

# Database imports (async)
try:
    from src.monitoring.db_manager import DatabaseManager
    from src.config import settings
    DB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Database imports failed: {e}. Running without database storage.")
    DB_AVAILABLE = False

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    logging.warning("Matplotlib not available. Plotting disabled.")
    PLOTTING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PROBLEM_SIZES = [10, 20, 30, 40, 50]
DEFAULT_EDGE_DENSITY = 0.5
DEFAULT_REPETITIONS = 3
QAOA_LAYERS = 1
QAOA_MAX_ITERATIONS = 50
QUANTUM_SHOTS = 1024
OUTPUT_DIR = project_root / "benchmark_results"
CSV_OUTPUT = OUTPUT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
REPORT_OUTPUT = OUTPUT_DIR / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_benchmark(
    problem_sizes: List[int] = DEFAULT_PROBLEM_SIZES,
    edge_density: float = DEFAULT_EDGE_DENSITY,
    repetitions: int = DEFAULT_REPETITIONS,
    save_to_db: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run comprehensive benchmark comparing classical and quantum solvers.
    
    For each problem size, this function:
    1. Generates random MaxCut problem instances
    2. Solves with classical solver (greedy + simulated annealing)
    3. Solves with quantum simulator (QAOA)
    4. Records all performance metrics
    5. Optionally saves to database
    6. Returns results as DataFrame
    
    Args:
        problem_sizes (List[int]): List of problem sizes to benchmark
        edge_density (float): Graph edge density (0.0 to 1.0)
        repetitions (int): Number of repetitions per size for statistical significance
        save_to_db (bool): Whether to save results to database
        verbose (bool): Whether to print progress information
    
    Returns:
        pd.DataFrame: Benchmark results with columns:
            - problem_size: Number of nodes
            - solver_type: 'classical_greedy', 'classical_sa', 'quantum'
            - execution_time_ms: Total runtime in milliseconds
            - energy_mj: Energy consumption in millijoules
            - cost: Objective function value (negative MaxCut value)
            - solution_quality: Quality score (0-1)
            - metadata: Additional solver-specific information
            - timestamp: When the benchmark was run
            - repetition: Repetition number (1 to repetitions)
    
    Example:
        >>> results = run_benchmark(problem_sizes=[10, 20, 30], repetitions=5)
        >>> print(results.groupby(['problem_size', 'solver_type'])['execution_time_ms'].mean())
    """
    logger.info("="*80)
    logger.info("QUANTUMEDGE PIPELINE - SOLVER BENCHMARK")
    logger.info("="*80)
    logger.info(f"Problem sizes: {problem_sizes}")
    logger.info(f"Edge density: {edge_density}")
    logger.info(f"Repetitions per size: {repetitions}")
    logger.info(f"Database storage: {save_to_db and DB_AVAILABLE}")
    logger.info("="*80 + "\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize solvers
    classical_solver = ClassicalSolver()
    quantum_solver = QuantumSimulator(shots=QUANTUM_SHOTS)
    
    # Results storage
    results = []
    
    # Total iterations for progress bar
    total_iterations = len(problem_sizes) * repetitions * 3  # 3 solver types
    
    with tqdm(total=total_iterations, desc="Benchmarking", disable=not verbose) as pbar:
        for size in problem_sizes:
            logger.info(f"\n{'='*80}")
            logger.info(f"PROBLEM SIZE: {size} nodes")
            logger.info(f"{'='*80}")
            
            for rep in range(1, repetitions + 1):
                if verbose:
                    logger.info(f"\nRepetition {rep}/{repetitions}")
                
                # Generate problem instance
                problem = MaxCutProblem(num_nodes=size)
                # problem.generate(edge_density=edge_density, seed=int(time.time() * 1000) + rep)
                problem.generate(edge_density=edge_density, seed=32)
                
                # Track problem metadata
                problem_metadata = {
                    "num_nodes": size,
                    "num_edges": problem.graph.number_of_edges(),
                    "edge_density": edge_density,
                    "graph_density": problem.get_metadata()['graph_density']
                }
                
                # ----------------------------------------------------------------
                # 1. Classical Solver - Greedy
                # ----------------------------------------------------------------
                if verbose:
                    logger.info("  [1/3] Running classical solver (greedy)...")
                
                try:
                    result_greedy = classical_solver.solve(
                        problem,
                        method='greedy',
                        verbose=False
                    )
                    
                    # Calculate solution quality (normalize by best known or upper bound)
                    greedy_quality = _estimate_solution_quality(
                        result_greedy['cost'],
                        problem,
                        result_greedy['solution']
                    )
                    
                    results.append({
                        'problem_size': size,
                        'solver_type': 'classical_greedy',
                        'execution_time_ms': result_greedy['time_ms'],
                        'energy_mj': result_greedy['energy_mj'],
                        'cost': result_greedy['cost'],
                        'solution_quality': greedy_quality,
                        'solution': result_greedy['solution'],
                        'metadata': {
                            **problem_metadata,
                            "method": 'greedy',
                            "iterations": result_greedy.get('iterations', 1)
                        },
                        'timestamp': datetime.now(),
                        'repetition': rep
                    })
                    
                    if verbose:
                        logger.info(f"     ✓ Greedy: {result_greedy['time_ms']}ms, "
                                  f"cost={result_greedy['cost']:.4f}, "
                                  f"quality={greedy_quality:.3f}")
                
                except Exception as e:
                    logger.error(f"     ✗ Greedy solver failed: {e}")
                
                pbar.update(1)
                
                # ----------------------------------------------------------------
                # 2. Classical Solver - Simulated Annealing
                # ----------------------------------------------------------------
                if verbose:
                    logger.info("  [2/3] Running classical solver (simulated annealing)...")
                
                try:
                    result_sa = classical_solver.solve(
                        problem,
                        method='simulated_annealing',
                        max_iterations=1000,
                        initial_temperature=100.0,
                        verbose=False
                    )
                    
                    sa_quality = _estimate_solution_quality(
                        result_sa['cost'],
                        problem,
                        result_sa['solution']
                    )
                    
                    results.append({
                        'problem_size': size,
                        'solver_type': 'classical_sa',
                        'execution_time_ms': result_sa['time_ms'],
                        'energy_mj': result_sa['energy_mj'],
                        'cost': result_sa['cost'],
                        'solution_quality': sa_quality,
                        'solution': result_sa['solution'],
                        'metadata': {
                            **problem_metadata,
                            "method": 'simulated_annealing',
                            "iterations": result_sa.get('iterations', 1000)
                        },
                        'timestamp': datetime.now(),
                        'repetition': rep
                    })
                    
                    if verbose:
                        logger.info(f"     ✓ Simulated Annealing: {result_sa['time_ms']}ms, "
                                  f"cost={result_sa['cost']:.4f}, "
                                  f"quality={sa_quality:.3f}")
                
                except Exception as e:
                    logger.error(f"     ✗ Simulated Annealing solver failed: {e}")
                
                pbar.update(1)
                
                # ----------------------------------------------------------------
                # 3. Quantum Solver - QAOA
                # ----------------------------------------------------------------
                if verbose:
                    logger.info("  [3/3] Running quantum solver (QAOA)...")
                
                try:
                    result_quantum = quantum_solver.solve(
                        problem,
                        p=QAOA_LAYERS,
                        maxiter=QAOA_MAX_ITERATIONS,
                        verbose=False
                    )
                    
                    quantum_quality = _estimate_solution_quality(
                        result_quantum['cost'],
                        problem,
                        result_quantum['solution']
                    )
                    
                    results.append({
                        'problem_size': size,
                        'solver_type': 'quantum',
                        'execution_time_ms': result_quantum['time_ms'],
                        'energy_mj': result_quantum['energy_mj'],
                        'cost': result_quantum['cost'],
                        'solution_quality': quantum_quality,
                        'solution': result_quantum['solution'],
                        'metadata': {
                            **problem_metadata,
                            "qaoa_layers": QAOA_LAYERS,
                            "max_iterations": QAOA_MAX_ITERATIONS,
                            "shots": QUANTUM_SHOTS,
                            "final_expectation": result_quantum['metadata'].get('final_expectation'),
                            "converged": result_quantum['metadata'].get('converged', False),
                            "iterations": result_quantum.get('iterations', 0)
                        },
                        'timestamp': datetime.now(),
                        'repetition': rep
                    })
                    
                    if verbose:
                        logger.info(f"     ✓ Quantum QAOA: {result_quantum['time_ms']}ms, "
                                  f"cost={result_quantum['cost']:.4f}, "
                                  f"quality={quantum_quality:.3f}")
                
                except Exception as e:
                    logger.error(f"     ✗ Quantum solver failed: {e}")
                
                pbar.update(1)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    df_results.to_csv(CSV_OUTPUT, index=False)
    logger.info(f"\n✓ Results saved to CSV: {CSV_OUTPUT}")
    
    # Save to database if enabled
    if save_to_db and DB_AVAILABLE:
        try:
            import asyncio
            asyncio.run(_save_to_database_async(results))
            logger.info("✓ Results saved to database")
        except Exception as e:
            logger.error(f"✗ Failed to save to database: {e}")
    
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Results saved to: {CSV_OUTPUT}")
    logger.info(f"{'='*80}\n")
    
    return df_results


def _estimate_solution_quality(
    cost: float,
    problem: MaxCutProblem,
    solution: List[int]
) -> float:
    """
    Estimate solution quality on a scale from 0 (worst) to 1 (optimal).
    
    For MaxCut, we use the approximation ratio against an upper bound.
    Since we don't know the optimal solution, we use:
    - Upper bound: Sum of all edge weights (all edges cut)
    - Lower bound: 0 (no edges cut)
    
    Quality = (|cost| - lower_bound) / (upper_bound - lower_bound)
    
    Args:
        cost (float): Objective function value (negative for MaxCut)
        problem (MaxCutProblem): Problem instance
        solution (List[int]): Solution bitstring
    
    Returns:
        float: Quality score between 0 and 1
    """
    # For MaxCut, cost is negative of cut value
    cut_value = abs(cost)
    
    # Upper bound: sum of all edge weights
    total_edge_weight = sum(
        data.get('weight', 1.0)
        for _, _, data in problem.graph.edges(data=True)
    )
    
    # Lower bound: 0 (no edges cut)
    lower_bound = 0.0
    upper_bound = total_edge_weight
    
    # Avoid division by zero
    if upper_bound == lower_bound:
        return 1.0
    
    # Calculate quality
    quality = (cut_value - lower_bound) / (upper_bound - lower_bound)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, quality))


async def _save_to_database_async(results: List[Dict[str, Any]]) -> None:
    """
    Save benchmark results to database asynchronously.
    
    Args:
        results (List[Dict[str, Any]]): Benchmark results to save
    """
    if not DB_AVAILABLE:
        return
    
    db_manager = DatabaseManager(database_url=settings.database.async_url)
    
    async with db_manager:
        for result in results:
            try:
                # Insert problem
                problem_id = await db_manager.insert_problem(
                    problem_type='maxcut',
                    problem_size=result['problem_size'],
                    graph_data=result['metadata']
                )
                
                # Map solver type to routing decision
                routing_map = {
                    'classical_greedy': 'classical',
                    'classical_sa': 'classical',
                    'quantum': 'quantum'
                }
                
                # Insert job execution
                await db_manager.insert_job_execution(
                    problem_id=problem_id,
                    routing_decision=routing_map[result['solver_type']],
                    routing_reason=f"Benchmark: {result['solver_type']}",
                    execution_time_ms=result['execution_time_ms'],
                    energy_consumed_mj=result['energy_mj'],
                    solution_quality=result['solution_quality'],
                    edge_profile='ground',  # Default profile for benchmarking
                    power_budget_used=50.0,  # Default value
                    solver_metadata=result['metadata']
                )
                
            except Exception as e:
                logger.error(f"Failed to save result to database: {e}")


# =============================================================================
# Results Analysis
# =============================================================================

def analyze_results(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Analyze benchmark results and identify quantum advantage patterns.
    
    This function performs comprehensive statistical analysis of benchmark results:
    1. Calculates average metrics by solver type and problem size
    2. Identifies problem sizes where quantum advantage emerges
    3. Computes speedup ratios and energy efficiency
    4. Analyzes solution quality trade-offs
    
    Args:
        csv_path (Optional[Path]): Path to CSV file with results.
                                   If None, uses most recent file in OUTPUT_DIR.
    
    Returns:
        Dict[str, Any]: Analysis results containing:
            - avg_time_by_solver: Average execution time per solver
            - avg_energy_by_solver: Average energy consumption per solver
            - avg_quality_by_solver: Average solution quality per solver
            - quantum_advantage_sizes: Problem sizes showing quantum advantage
            - speedup_ratios: Quantum speedup for each problem size
            - best_solver_by_size: Recommended solver for each problem size
            - summary_statistics: Overall performance summary
    
    Example:
        >>> analysis = analyze_results()
        >>> print(f"Quantum advantage at sizes: {analysis['quantum_advantage_sizes']}")
        >>> print(f"Best solver for size 30: {analysis['best_solver_by_size'][30]}")
    """
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK ANALYSIS")
    logger.info("="*80)
    
    # Load results
    if csv_path is None:
        # Find most recent CSV
        csv_files = sorted(OUTPUT_DIR.glob("benchmark_*.csv"))
        if not csv_files:
            logger.error("No benchmark results found!")
            return {}
        csv_path = csv_files[-1]
    
    logger.info(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # # Convert metadata from string to dict if needed
    # if 'metadata' in df.columns and isinstance(df['metadata'].iloc[0], str):
    #     df['metadata'] = df['metadata'].replace('\'', '"')
    #     df['metadata'] = df['metadata'].apply(json.loads)
    
    analysis = {}
    
    # -------------------------------------------------------------------------
    # 1. Average Time by Solver and Problem Size
    # -------------------------------------------------------------------------
    logger.info("\n1. Average Execution Time (ms)")
    logger.info("-" * 80)
    
    avg_time = df.groupby(['problem_size', 'solver_type'])['execution_time_ms'].mean().unstack()
    analysis['avg_time_by_solver'] = avg_time
    
    for size in sorted(df['problem_size'].unique()):
        logger.info(f"  Size {size:3d}: ")
        for solver in ['classical_greedy', 'classical_sa', 'quantum']:
            if solver in avg_time.columns:
                time_val = avg_time.loc[size, solver]
                logger.info(f"{solver:20s}: {time_val:8.2f}ms  ")
        logger.info("")
    
    # -------------------------------------------------------------------------
    # 2. Average Energy by Solver and Problem Size
    # -------------------------------------------------------------------------
    logger.info("\n2. Average Energy Consumption (mJ)")
    logger.info("-" * 80)
    
    avg_energy = df.groupby(['problem_size', 'solver_type'])['energy_mj'].mean().unstack()
    analysis['avg_energy_by_solver'] = avg_energy
    
    for size in sorted(df['problem_size'].unique()):
        logger.info(f"  Size {size:3d}: ")
        for solver in ['classical_greedy', 'classical_sa', 'quantum']:
            if solver in avg_energy.columns:
                energy_val = avg_energy.loc[size, solver]
                logger.info(f"{solver:20s}: {energy_val:8.2f}mJ  ")
        logger.info("")
    
    # -------------------------------------------------------------------------
    # 3. Average Quality by Solver and Problem Size
    # -------------------------------------------------------------------------
    logger.info("\n3. Average Solution Quality (0-1)")
    logger.info("-" * 80)
    
    avg_quality = df.groupby(['problem_size', 'solver_type'])['solution_quality'].mean().unstack()
    analysis['avg_quality_by_solver'] = avg_quality
    
    for size in sorted(df['problem_size'].unique()):
        logger.info(f"  Size {size:3d}: ")
        for solver in ['classical_greedy', 'classical_sa', 'quantum']:
            if solver in avg_quality.columns:
                quality_val = avg_quality.loc[size, solver]
                logger.info(f"{solver:20s}: {quality_val:6.4f}  ")
        logger.info("")
    
    # -------------------------------------------------------------------------
    # 4. Quantum Advantage Analysis
    # -------------------------------------------------------------------------
    logger.info("\n4. Quantum Advantage Analysis")
    logger.info("-" * 80)
    
    quantum_advantage_sizes = []
    speedup_ratios = {}
    
    for size in sorted(df['problem_size'].unique()):
        # Get average times for this size
        quantum_time = avg_time.loc[size, 'quantum'] if 'quantum' in avg_time.columns else None
        greedy_time = avg_time.loc[size, 'classical_greedy'] if 'classical_greedy' in avg_time.columns else None
        sa_time = avg_time.loc[size, 'classical_sa'] if 'classical_sa' in avg_time.columns else None
        
        if quantum_time and greedy_time:
            speedup_vs_greedy = greedy_time / quantum_time
            speedup_ratios[f'{size}_vs_greedy'] = speedup_vs_greedy
            
            if speedup_vs_greedy > 1.0:
                quantum_advantage_sizes.append(size)
                logger.info(f"  ✓ Size {size:3d}: Quantum advantage vs Greedy! "
                          f"Speedup: {speedup_vs_greedy:.2f}x")
            else:
                logger.info(f"  ✗ Size {size:3d}: No advantage vs Greedy. "
                          f"Ratio: {speedup_vs_greedy:.2f}x")
        
        if quantum_time and sa_time:
            speedup_vs_sa = sa_time / quantum_time
            speedup_ratios[f'{size}_vs_sa'] = speedup_vs_sa
            
            if speedup_vs_sa > 1.0:
                logger.info(f"  ✓ Size {size:3d}: Quantum advantage vs SA! "
                          f"Speedup: {speedup_vs_sa:.2f}x")
    
    analysis['quantum_advantage_sizes'] = quantum_advantage_sizes
    analysis['speedup_ratios'] = speedup_ratios
    
    # -------------------------------------------------------------------------
    # 5. Best Solver Recommendation by Problem Size
    # -------------------------------------------------------------------------
    logger.info("\n5. Best Solver by Problem Size")
    logger.info("-" * 80)
    
    best_solver_by_size = {}
    
    for size in sorted(df['problem_size'].unique()):
        # Find solver with best time-quality trade-off
        # Metric: quality / time (higher is better)
        scores = {}
        
        for solver in ['classical_greedy', 'classical_sa', 'quantum']:
            if solver in avg_time.columns and solver in avg_quality.columns:
                time_val = avg_time.loc[size, solver]
                quality_val = avg_quality.loc[size, solver]
                
                # Trade-off score: quality per millisecond
                scores[solver] = quality_val / (time_val + 1)  # +1 to avoid division by zero
        
        if scores:
            best_solver = max(scores, key=scores.get)
            best_solver_by_size[size] = best_solver
            logger.info(f"  Size {size:3d}: {best_solver:20s} "
                      f"(score: {scores[best_solver]:.6f})")
    
    analysis['best_solver_by_size'] = best_solver_by_size
    
    # -------------------------------------------------------------------------
    # 6. Summary Statistics
    # -------------------------------------------------------------------------
    logger.info("\n6. Summary Statistics")
    logger.info("-" * 80)
    
    summary = {
        'total_experiments': len(df),
        'problem_sizes_tested': sorted(df['problem_size'].unique().tolist()),
        'solvers_tested': df['solver_type'].unique().tolist(),
        'quantum_advantage_observed': len(quantum_advantage_sizes) > 0,
        'optimal_size_range': quantum_advantage_sizes if quantum_advantage_sizes else None
    }
    
    logger.info(f"  Total experiments: {summary['total_experiments']}")
    logger.info(f"  Problem sizes: {summary['problem_sizes_tested']}")
    logger.info(f"  Solvers: {summary['solvers_tested']}")
    logger.info(f"  Quantum advantage observed: {summary['quantum_advantage_observed']}")
    if summary['optimal_size_range']:
        logger.info(f"  Quantum optimal for sizes: {summary['optimal_size_range']}")
    
    analysis['summary_statistics'] = summary
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80 + "\n")
    
    return analysis


# =============================================================================
# Visualization
# =============================================================================

def plot_performance(csv_path: Optional[Path] = None) -> None:
    """
    Create comprehensive performance visualization plots.
    
    Generates four plots:
    1. Execution Time vs Problem Size (log scale)
    2. Energy Consumption vs Problem Size
    3. Solution Quality vs Problem Size
    4. Quality vs Time Trade-off
    
    Args:
        csv_path (Optional[Path]): Path to CSV file with results.
                                   If None, uses most recent file in OUTPUT_DIR.
    
    Example:
        >>> plot_performance()  # Uses most recent results
        >>> plot_performance(Path("benchmark_results/benchmark_20240115_143022.csv"))
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping plots.")
        return
    
    logger.info("\n" + "="*80)
    logger.info("GENERATING PERFORMANCE PLOTS")
    logger.info("="*80)
    
    # Load results
    if csv_path is None:
        csv_files = sorted(OUTPUT_DIR.glob("benchmark_*.csv"))
        if not csv_files:
            logger.error("No benchmark results found!")
            return
        csv_path = csv_files[-1]
    
    logger.info(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Calculate averages
    avg_metrics = df.groupby(['problem_size', 'solver_type']).agg({
        'execution_time_ms': 'mean',
        'energy_mj': 'mean',
        'solution_quality': 'mean'
    }).reset_index()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QuantumEdge Pipeline - Solver Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = {
        'classical_greedy': '#2E86AB',
        'classical_sa': '#A23B72',
        'quantum': '#F18F01'
    }
    
    markers = {
        'classical_greedy': 'o',
        'classical_sa': 's',
        'quantum': '^'
    }
    
    # -------------------------------------------------------------------------
    # Plot 1: Execution Time vs Problem Size (log scale)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    
    for solver in ['classical_greedy', 'classical_sa', 'quantum']:
        data = avg_metrics[avg_metrics['solver_type'] == solver]
        ax1.plot(data['problem_size'], data['execution_time_ms'],
                marker=markers[solver], color=colors[solver], linewidth=2,
                markersize=8, label=solver.replace('_', ' ').title())
    
    ax1.set_xlabel('Problem Size (number of nodes)', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('Execution Time vs Problem Size', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Energy Consumption vs Problem Size
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    for solver in ['classical_greedy', 'classical_sa', 'quantum']:
        data = avg_metrics[avg_metrics['solver_type'] == solver]
        ax2.plot(data['problem_size'], data['energy_mj'],
                marker=markers[solver], color=colors[solver], linewidth=2,
                markersize=8, label=solver.replace('_', ' ').title())
    
    ax2.set_xlabel('Problem Size (number of nodes)', fontsize=12)
    ax2.set_ylabel('Energy Consumption (mJ)', fontsize=12)
    ax2.set_title('Energy Consumption vs Problem Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Solution Quality vs Problem Size
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    for solver in ['classical_greedy', 'classical_sa', 'quantum']:
        data = avg_metrics[avg_metrics['solver_type'] == solver]
        ax3.plot(data['problem_size'], data['solution_quality'],
                marker=markers[solver], color=colors[solver], linewidth=2,
                markersize=8, label=solver.replace('_', ' ').title())
    
    ax3.set_xlabel('Problem Size (number of nodes)', fontsize=12)
    ax3.set_ylabel('Solution Quality (0-1)', fontsize=12)
    ax3.set_title('Solution Quality vs Problem Size', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 4: Quality vs Time Trade-off
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    for solver in ['classical_greedy', 'classical_sa', 'quantum']:
        data = avg_metrics[avg_metrics['solver_type'] == solver]
        ax4.scatter(data['execution_time_ms'], data['solution_quality'],
                   marker=markers[solver], color=colors[solver], s=100,
                   label=solver.replace('_', ' ').title(), alpha=0.7)
        
        # Add size annotations
        for _, row in data.iterrows():
            ax4.annotate(f"{int(row['problem_size'])}", 
                        (row['execution_time_ms'], row['solution_quality']),
                        fontsize=8, alpha=0.7)
    
    ax4.set_xlabel('Execution Time (ms)', fontsize=12)
    ax4.set_ylabel('Solution Quality (0-1)', fontsize=12)
    ax4.set_title('Quality vs Time Trade-off', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / f"performance_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Plots saved to: {plot_path}")
    
    plt.close()
    
    logger.info("="*80 + "\n")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(csv_path: Optional[Path] = None, analysis: Optional[Dict] = None) -> None:
    """
    Generate comprehensive Markdown report with benchmark findings.
    
    Creates a detailed report including:
    - Executive summary
    - Performance metrics by solver
    - Quantum advantage analysis
    - Recommendations for solver selection
    - Raw data tables
    
    Args:
        csv_path (Optional[Path]): Path to CSV file with results
        analysis (Optional[Dict]): Pre-computed analysis results
    
    Example:
        >>> generate_report()  # Auto-generates from most recent results
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING BENCHMARK REPORT")
    logger.info("="*80)
    
    # Load results
    if csv_path is None:
        csv_files = sorted(OUTPUT_DIR.glob("benchmark_*.csv"))
        if not csv_files:
            logger.error("No benchmark results found!")
            return
        csv_path = csv_files[-1]
    
    # Run analysis if not provided
    if analysis is None:
        analysis = analyze_results(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Generate report content
    report = []
    report.append("# QuantumEdge Pipeline - Benchmark Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Data Source:** `{csv_path.name}`")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    summary = analysis.get('summary_statistics', {})
    report.append(f"- **Total Experiments:** {summary.get('total_experiments', 0)}")
    report.append(f"- **Problem Sizes Tested:** {summary.get('problem_sizes_tested', [])}")
    report.append(f"- **Solvers Evaluated:** {summary.get('solvers_tested', [])}")
    report.append(f"- **Quantum Advantage Observed:** {summary.get('quantum_advantage_observed', False)}")
    
    if summary.get('optimal_size_range'):
        report.append(f"- **Quantum Optimal for Sizes:** {summary['optimal_size_range']}")
    
    report.append("\n---\n")
    
    # Performance Metrics
    report.append("## Performance Metrics\n")
    
    report.append("### Execution Time (ms)\n")
    if 'avg_time_by_solver' in analysis:
        report.append(analysis['avg_time_by_solver'].to_markdown())
    report.append("\n")
    
    report.append("### Energy Consumption (mJ)\n")
    if 'avg_energy_by_solver' in analysis:
        report.append(analysis['avg_energy_by_solver'].to_markdown())
    report.append("\n")
    
    report.append("### Solution Quality (0-1)\n")
    if 'avg_quality_by_solver' in analysis:
        report.append(analysis['avg_quality_by_solver'].to_markdown())
    report.append("\n")
    
    report.append("\n---\n")
    
    # Quantum Advantage
    report.append("## Quantum Advantage Analysis\n")
    
    if analysis.get('quantum_advantage_sizes'):
        report.append(f"Quantum solver shows advantage for problem sizes: "
                     f"**{analysis['quantum_advantage_sizes']}**\n")
    else:
        report.append("No quantum advantage observed in tested problem sizes.\n")
    
    report.append("\n### Speedup Ratios\n")
    speedup = analysis.get('speedup_ratios', {})
    for key, value in speedup.items():
        size, vs = key.split('_vs_')
        report.append(f"- Size {size} vs {vs}: **{value:.2f}x**")
    
    report.append("\n\n---\n")
    
    # Recommendations
    report.append("## Solver Selection Recommendations\n")
    
    if 'best_solver_by_size' in analysis:
        report.append("| Problem Size | Recommended Solver | Reason |\n")
        report.append("|--------------|-------------------|--------|\n")
        
        for size, solver in analysis['best_solver_by_size'].items():
            reason = "Best time-quality trade-off"
            if size in analysis.get('quantum_advantage_sizes', []):
                reason += " (quantum advantage)"
            report.append(f"| {size} | {solver} | {reason} |\n")
    
    report.append("\n---\n")
    
    # Conclusions
    report.append("## Conclusions\n")
    
    if analysis.get('quantum_advantage_sizes'):
        report.append(f"1. **Quantum advantage emerges** for problem sizes {analysis['quantum_advantage_sizes']}\n")
        report.append("2. For these sizes, QAOA provides faster solutions with comparable quality\n")
    else:
        report.append("1. **No quantum advantage** observed in the tested problem size range\n")
        report.append("2. Classical solvers (especially greedy) remain competitive for small problems\n")
    
    report.append("3. Simulated annealing provides highest quality but at computational cost\n")
    report.append("4. Greedy algorithms offer fastest execution for quick approximations\n")
    
    report.append("\n---\n")
    report.append("\n*Report generated by QuantumEdge Pipeline Benchmarking System*\n")
    
    # Write report to file
    with open(REPORT_OUTPUT, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✓ Report saved to: {REPORT_OUTPUT}")
    logger.info("="*80 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for benchmark script.
    
    Parses command-line arguments and runs benchmark workflow:
    1. Run benchmark experiments
    2. Analyze results
    3. Generate plots (if requested)
    4. Generate report
    """
    parser = argparse.ArgumentParser(
        description='Benchmark QuantumEdge Pipeline solvers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=DEFAULT_PROBLEM_SIZES,
        help='Problem sizes to benchmark'
    )
    
    parser.add_argument(
        '--repetitions',
        type=int,
        default=DEFAULT_REPETITIONS,
        help='Number of repetitions per size'
    )
    
    parser.add_argument(
        '--edge-density',
        type=float,
        default=DEFAULT_EDGE_DENSITY,
        help='Graph edge density (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate performance plots'
    )
    
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Disable database storage'
    )
    
    parser.add_argument(
        '--csv',
        type=Path,
        help='Path to existing CSV for analysis only (skip benchmark)'
    )
    
    args = parser.parse_args()
    
    # Run workflow
    if args.csv:
        # Analysis only mode
        logger.info("Running analysis on existing results...")
        analysis = analyze_results(args.csv)
        
        if args.plot:
            plot_performance(args.csv)
        
        generate_report(args.csv, analysis)
    else:
        # Full benchmark mode
        logger.info("Running full benchmark...")
        
        # Run benchmark
        results_df = run_benchmark(
            problem_sizes=args.sizes,
            edge_density=args.edge_density,
            repetitions=args.repetitions,
            save_to_db=not args.no_db,
            verbose=True
        )
        
        # Analyze results
        analysis = analyze_results(CSV_OUTPUT)
        
        # Generate plots
        if args.plot:
            plot_performance(CSV_OUTPUT)
        
        # Generate report
        generate_report(CSV_OUTPUT, analysis)
    
    logger.info("✓ Benchmark workflow complete!")


if __name__ == '__main__':
    main()
