"""
Dashboard Visualization Utilities

Reusable visualization functions for the QuantumEdge Pipeline dashboard.
All functions return interactive Plotly figures for integration with Streamlit.

Functions:
---------
- plot_graph_solution: Visualize graph problems with solution coloring
- plot_performance_comparison: Side-by-side classical vs quantum comparison
- plot_routing_decision_flow: Flowchart of routing decision process
- plot_historical_trends: Time series analysis of multiple metrics
- create_performance_heatmap: Problem size vs solver type performance matrix

All visualizations are interactive with zoom, pan, and hover capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from datetime import datetime


def plot_graph_solution(problem, solution: List[int]) -> go.Figure:
    """
    Visualize graph problem with solution coloring.
    
    For MaxCut problems, colors nodes by partition and highlights cut edges.
    For TSP problems, shows the tour path.
    For Portfolio problems, shows asset allocation.
    
    Parameters
    ----------
    problem : Problem
        The problem instance (MaxCut, TSP, or Portfolio)
    solution : Dict[str, Any]
        Solution dictionary containing the result
    
    Returns
    -------
    go.Figure
        Interactive Plotly figure with the solution visualization
    
    Example
    -------
    >>> from src.problems.maxcut import MaxCutProblem
    >>> problem = MaxCutProblem(num_nodes=20)
    >>> problem.generate(edge_probability=0.3)
    >>> solution = {'partition': [0, 1, 0, 1, ...]}
    >>> fig = plot_graph_solution(problem, solution)
    >>> fig.show()
    """
    problem_type = problem.__class__.__name__

    if problem_type == "MaxCutProblem":
        return _plot_maxcut_solution(problem, solution)
    elif problem_type == "TSPProblem":
        return _plot_tsp_solution(problem, solution)
    elif problem_type == "PortfolioOptimizationProblem":
        return _plot_portfolio_solution(problem, solution)
    else:
        # Fallback: simple text display
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization not implemented for {problem_type}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig


def _plot_maxcut_solution(problem, solution: List[int]) -> go.Figure:
    """Plot MaxCut problem solution with colored partitions."""
    # Build NetworkX graph from problem
    G = nx.Graph()
    
    # Add nodes
    num_nodes = problem.num_nodes
    G.add_nodes_from(range(num_nodes))
    
    # Add edges from adjacency matrix
    adj_matrix = problem.adjacency_matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i][j])
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, k=2/np.sqrt(num_nodes), iterations=50, seed=42)
    
    # Get partition from solution
    partition = solution if len(solution) == 0 else  [0] * num_nodes
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Check if this is a cut edge
        is_cut = partition[edge[0]] != partition[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=2 if is_cut else 1,
                color='red' if is_cut else 'lightgray'
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node traces by partition
    node_traces = []
    colors = ['#FF6B6B', '#4ECDC4']  # Red and Teal
    partition_names = ['Partition 0', 'Partition 1']
    
    for part_id in [0, 1]:
        node_indices = [i for i, p in enumerate(partition) if p == part_id]
        
        if not node_indices:
            continue
        
        node_x = [pos[i][0] for i in node_indices]
        node_y = [pos[i][1] for i in node_indices]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            name=partition_names[part_id],
            marker=dict(
                size=20,
                color=colors[part_id],
                line=dict(width=2, color='white')
            ),
            text=[str(i) for i in node_indices],
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            hovertemplate='Node %{text}<br>Partition: ' + str(part_id) + '<extra></extra>'
        )
        node_traces.append(node_trace)
    
    # Combine traces
    fig = go.Figure(data=edge_traces + node_traces)
    
    # Count cut edges
    cut_count = sum(
        1 for i, j in G.edges()
        if partition[i] != partition[j]
    )
    
    # Update layout
    fig.update_layout(
        title=f"MaxCut Solution: {cut_count} Cut Edges (Red)",
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )
    
    return fig


def _plot_tsp_solution(problem, solution: List[int]) -> go.Figure:
    """Plot TSP problem solution with tour path."""
    # Get tour from solution
    tour = solution if len(solution) == 0 else  list(range(problem.num_cities))
    
    # Get city positions
    positions = problem.coordinates
    
    # Create figure
    fig = go.Figure()
    
    # Add tour path
    tour_x = [positions[i][0] for i in tour] + [positions[tour[0]][0]]
    tour_y = [positions[i][1] for i in tour] + [positions[tour[0]][1]]
    
    fig.add_trace(go.Scatter(
        x=tour_x,
        y=tour_y,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Tour Path',
        hoverinfo='skip'
    ))
    
    # Add cities
    fig.add_trace(go.Scatter(
        x=[pos[0] for pos in positions],
        y=[pos[1] for pos in positions],
        mode='markers+text',
        marker=dict(size=12, color='red', line=dict(width=2, color='white')),
        text=[str(i) for i in range(problem.num_cities)],
        textposition='top center',
        name='Cities',
        hovertemplate='City %{text}<extra></extra>'
    ))
    
    # Calculate total distance
    total_distance = problem.calculate_cost(tour)
    
    fig.update_layout(
        title=f"TSP Solution: Total Distance = {total_distance:.2f}",
        showlegend=True,
        xaxis=dict(title="X Coordinate"),
        yaxis=dict(title="Y Coordinate"),
        plot_bgcolor='white',
        height=600
    )
    
    return fig


def _plot_portfolio_solution(problem, solution: List[int]) -> go.Figure:
    """Plot Portfolio problem solution with asset allocation."""
    # Get allocation from solution
    allocation = solution if len(solution) == 0 else  [0] * problem.num_assets
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"Asset {i}" for i in range(problem.num_assets)],
        y=allocation,
        marker=dict(
            color=allocation,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Weight")
        ),
        hovertemplate='%{x}<br>Weight: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Allocation",
        xaxis=dict(title="Assets"),
        yaxis=dict(title="Weight"),
        plot_bgcolor='white',
        height=500
    )
    
    return fig


def plot_performance_comparison(
    classical_result: Dict[str, Any],
    quantum_result: Dict[str, Any]
) -> go.Figure:
    """
    Create side-by-side comparison of classical vs quantum results.
    
    Compares execution time, energy consumption, and solution quality
    with color coding (green = better).
    
    Parameters
    ----------
    classical_result : Dict[str, Any]
        Results from classical solver
    quantum_result : Dict[str, Any]
        Results from quantum solver
    
    Returns
    -------
    go.Figure
        Interactive Plotly bar chart with comparison
    
    Example
    -------
    >>> classical = {'time_ms': 150, 'energy_mj': 12.5, 'quality': 0.85}
    >>> quantum = {'time_ms': 200, 'energy_mj': 8.0, 'quality': 0.90}
    >>> fig = plot_performance_comparison(classical, quantum)
    >>> fig.show()
    """
    # Extract metrics
    metrics = ['Time (ms)', 'Energy (mJ)', 'Quality (%)']
    
    classical_values = [
        classical_result.get('time_ms', 0),
        classical_result.get('energy_mj', 0),
        classical_result.get('quality', 0) * 100
    ]
    
    quantum_values = [
        quantum_result.get('time_ms', 0),
        quantum_result.get('energy_mj', 0),
        quantum_result.get('quality', 0) * 100
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=metrics,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Determine winners (lower is better for time and energy, higher for quality)
    time_winner = 'quantum' if quantum_values[0] < classical_values[0] else 'classical'
    energy_winner = 'quantum' if quantum_values[1] < classical_values[1] else 'classical'
    quality_winner = 'quantum' if quantum_values[2] > classical_values[2] else 'classical'
    
    # Time comparison
    fig.add_trace(
        go.Bar(
            x=['Classical', 'Quantum'],
            y=[classical_values[0], quantum_values[0]],
            marker_color=['green' if time_winner == 'classical' else 'lightblue',
                         'green' if time_winner == 'quantum' else 'lightcoral'],
            text=[f"{classical_values[0]:.0f}", f"{quantum_values[0]:.0f}"],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Energy comparison
    fig.add_trace(
        go.Bar(
            x=['Classical', 'Quantum'],
            y=[classical_values[1], quantum_values[1]],
            marker_color=['green' if energy_winner == 'classical' else 'lightblue',
                         'green' if energy_winner == 'quantum' else 'lightcoral'],
            text=[f"{classical_values[1]:.2f}", f"{quantum_values[1]:.2f}"],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Quality comparison
    fig.add_trace(
        go.Bar(
            x=['Classical', 'Quantum'],
            y=[classical_values[2], quantum_values[2]],
            marker_color=['green' if quality_winner == 'classical' else 'lightblue',
                         'green' if quality_winner == 'quantum' else 'lightcoral'],
            text=[f"{classical_values[2]:.1f}%", f"{quantum_values[2]:.1f}%"],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="Classical vs Quantum Performance (Green = Better)",
        showlegend=False,
        height=400
    )
    
    return fig


def plot_routing_decision_flow(routing_result: Dict[str, Any]) -> go.Figure:
    """
    Create flowchart showing routing decision process.
    
    Visualizes the decision tree from problem analysis through constraints
    to final solver selection using a Sankey diagram.
    
    Parameters
    ----------
    routing_result : Dict[str, Any]
        Routing decision information including constraints and reasoning
    
    Returns
    -------
    go.Figure
        Interactive Plotly Sankey diagram
    
    Example
    -------
    >>> routing = {
    ...     'chosen_solver': 'quantum',
    ...     'constraints': ['power_budget', 'latency'],
    ...     'reasoning': 'Quantum advantages for this problem size'
    ... }
    >>> fig = plot_routing_decision_flow(routing)
    >>> fig.show()
    """
    # Extract information
    chosen_solver = routing_result.get('chosen_solver', 'unknown')
    constraints = routing_result.get('constraints', [])
    reasoning = routing_result.get('reasoning', '')
    
    # Build Sankey diagram nodes
    nodes = ['Problem', 'Analysis', 'Constraints', 'Decision', chosen_solver.title()]
    node_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'green']
    
    # Build links
    source = [0, 1, 2, 3]  # From indices
    target = [1, 2, 3, 4]  # To indices
    values = [1, 1, 1, 1]  # Flow values
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color='rgba(0, 0, 0, 0.2)'
        )
    )])
    
    fig.update_layout(
        title=f"Routing Decision Flow → {chosen_solver.title()}",
        font=dict(size=12),
        height=400
    )
    
    return fig


def plot_historical_trends(metrics_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create time series visualization with multiple metrics.
    
    Plots execution time, energy consumption, and solution quality over time
    with support for multiple Y-axes and interactive zoom/pan.
    
    Parameters
    ----------
    metrics_data : List[Dict[str, Any]]
        List of job results with metrics
    
    Returns
    -------
    go.Figure
        Interactive Plotly figure with multiple time series
    
    Example
    -------
    >>> history = [
    ...     {'time_ms': 150, 'energy_mj': 10, 'quality': 0.85, 'completed_at': datetime.now()},
    ...     {'time_ms': 120, 'energy_mj': 8, 'quality': 0.90, 'completed_at': datetime.now()}
    ... ]
    >>> fig = plot_historical_trends(history)
    >>> fig.show()
    """
    if not metrics_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Extract timestamps and metrics
    timestamps = [r.get('completed_at', datetime.now()) for r in metrics_data]
    times = [r.get('time_ms', 0) for r in metrics_data]
    energies = [r.get('energy_mj', 0) for r in metrics_data]
    qualities = [r.get('solution_quality', r.get('quality', 0)) * 100 for r in metrics_data]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add execution time trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=times,
            name="Execution Time (ms)",
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # Add energy trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=energies,
            name="Energy (mJ)",
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # Add quality trace on secondary axis
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=qualities,
            name="Quality (%)",
            mode='lines+markers',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Time (ms) / Energy (mJ)", secondary_y=False)
    fig.update_yaxes(title_text="Quality (%)", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title="Performance Trends Over Time",
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_performance_heatmap(
    size_range: List[int],
    solver_types: List[str],
    performance_data: Optional[Dict[Tuple[int, str], float]] = None
) -> go.Figure:
    """
    Create heatmap of performance across problem sizes and solver types.
    
    Color represents execution time, with quantum advantage zones highlighted.
    
    Parameters
    ----------
    size_range : List[int]
        Range of problem sizes
    solver_types : List[str]
        List of solver types (e.g., ['classical', 'quantum'])
    performance_data : Optional[Dict[Tuple[int, str], float]]
        Performance data as {(size, solver): time_ms}
        If None, generates dummy data for demonstration
    
    Returns
    -------
    go.Figure
        Interactive Plotly heatmap
    
    Example
    -------
    >>> sizes = [10, 20, 30, 40, 50]
    >>> solvers = ['classical', 'quantum']
    >>> data = {(10, 'classical'): 50, (10, 'quantum'): 80, ...}
    >>> fig = create_performance_heatmap(sizes, solvers, data)
    >>> fig.show()
    """
    # Generate dummy data if not provided
    if performance_data is None:
        performance_data = {}
        for size in size_range:
            # Classical grows linearly
            performance_data[(size, 'classical')] = size * 2
            # Quantum has advantage at larger sizes
            performance_data[(size, 'quantum')] = size * 3 if size < 30 else size * 1.5
    
    # Build matrix
    matrix = []
    for solver in solver_types:
        row = [performance_data.get((size, solver), 0) for size in size_range]
        matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=size_range,
        y=solver_types,
        colorscale='RdYlGn_r',  # Red (slow) to Green (fast)
        text=[[f"{val:.1f} ms" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Time (ms)")
    ))
    
    fig.update_layout(
        title="Performance Heatmap: Problem Size × Solver Type",
        xaxis=dict(title="Problem Size"),
        yaxis=dict(title="Solver Type"),
        height=400
    )
    
    return fig
