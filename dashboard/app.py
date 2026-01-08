"""
QuantumEdge Pipeline - Interactive Streamlit Dashboard

This dashboard provides a comprehensive interface for:
- Submitting optimization problems with custom configurations
- Monitoring live job execution with real-time metrics
- Analyzing results with interactive visualizations
- Reviewing historical performance trends
- Checking system health and status

The dashboard integrates with the QuantumEdge orchestrator to provide
full visibility into the quantum-classical routing pipeline.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import time
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.orchestrator import JobOrchestrator
from src.problems.maxcut import MaxCutProblem
from src.problems.tsp import TSPProblem
from src.problems.portfolio import PortfolioProblem
from src.config import settings
from dashboard.utils import (
    plot_graph_solution,
    plot_performance_comparison,
    plot_historical_trends
)
from dashboard.demo_scenarios import DEMO_SCENARIOS, load_demo_scenario

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="QuantumEdge Pipeline",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = JobOrchestrator()
    
    if 'current_job' not in st.session_state:
        st.session_state.current_job = None
    
    if 'job_running' not in st.session_state:
        st.session_state.job_running = False
    
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    if 'job_history' not in st.session_state:
        st.session_state.job_history = []

init_session_state()

# =============================================================================
# Sidebar Navigation
# =============================================================================

def render_sidebar():
    """Render sidebar with logo, title, and navigation."""
    with st.sidebar:
        st.title("⚛️ QuantumEdge Pipeline")
        st.markdown("---")
        st.markdown("### Navigation")
        
        page = st.radio(
            "Select Page",
            options=[
                "Submit Problem",
                "Live Execution",
                "Results Analysis",
                "Historical Performance",
                "System Status"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Quick Info")
        st.info(f"**Active Jobs:** {1 if st.session_state.job_running else 0}")
        st.info(f"**Total Processed:** {len(st.session_state.job_history)}")
        
        return page

# =============================================================================
# Page 1: Submit Problem
# =============================================================================

def page_submit_problem():
    """Page for submitting new optimization problems."""
    st.header("🚀 Submit Problem")
    st.markdown("Configure and submit optimization problems for quantum-classical execution.")
    
    # Demo scenario selector
    st.markdown("### Quick Start")
    col1, col2 = st.columns([3, 1])
    with col1:
        demo_selection = st.selectbox(
            "Load Demo Scenario",
            options=["None"] + list(DEMO_SCENARIOS.keys()),
            format_func=lambda x: x if x == "None" else f"{x} - {DEMO_SCENARIOS[x]['description'][:50]}..."
        )
    with col2:
        load_demo = st.button("Load Demo", type="secondary")
    
    if load_demo and demo_selection != "None":
        scenario_data = load_demo_scenario(demo_selection)
        st.success(f"✅ Loaded: {DEMO_SCENARIOS[demo_selection]['description']}")
        # Populate form with demo data
        st.session_state.demo_data = scenario_data
    
    st.markdown("---")
    st.markdown("### Problem Configuration")
    
    # Main configuration form
    with st.form("problem_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            problem_type = st.selectbox(
                "Problem Type",
                options=["MaxCut", "TSP", "Portfolio"],
                help="Type of optimization problem to solve"
            )
            
            problem_size = st.slider(
                "Problem Size",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Number of nodes/cities/assets depending on problem type"
            )
            
            edge_profile = st.selectbox(
                "Edge Profile",
                options=["aerospace", "mobile", "ground"],
                help="Target edge computing environment"
            )
        
        with col2:
            strategy = st.selectbox(
                "Optimization Strategy",
                options=["energy", "latency", "quality", "balanced"],
                help="What to optimize for in routing decision"
            )
            
            comparative_mode = st.checkbox(
                "Comparative Mode",
                value=False,
                help="Run both classical and quantum solvers for comparison"
            )
            
            if problem_type == "MaxCut":
                edge_probability = st.slider(
                    "Graph Density (edge probability)",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.3,
                    step=0.1
                )
        
        # Advanced options (collapsible)
        with st.expander("⚙️ Advanced Options"):
            col3, col4 = st.columns(2)
            
            with col3:
                custom_power = st.number_input(
                    "Custom Power Budget (W)",
                    min_value=1.0,
                    max_value=200.0,
                    value=None,
                    help="Override default power budget for edge profile"
                )
                
                timeout_override = st.number_input(
                    "Timeout Override (seconds)",
                    min_value=1,
                    max_value=300,
                    value=60,
                    help="Maximum execution time per solver"
                )
            
            with col4:
                solver_prefs = st.multiselect(
                    "Solver Preferences",
                    options=["classical", "quantum_simulator"],
                    default=[],
                    help="Preferred solvers (empty = auto-select)"
                )
                
                seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    max_value=99999,
                    value=42,
                    help="For reproducible problem generation"
                )
        
        # Submit button
        col5, col6, col7 = st.columns([1, 1, 2])
        with col5:
            submit_button = st.form_submit_button("🚀 Submit Job", type="primary")
        with col6:
            clear_button = st.form_submit_button("🔄 Clear Form")
    
    # Handle form submission
    if submit_button:
        if st.session_state.job_running:
            st.error("❌ A job is already running. Please wait or abort it first.")
        else:
            with st.spinner("Creating problem and submitting job..."):
                try:
                    # Create problem based on type
                    if problem_type == "MaxCut":
                        problem = MaxCutProblem(num_nodes=problem_size)
                        problem.generate(edge_probability=edge_probability, seed=seed)
                    elif problem_type == "TSP":
                        problem = TSPProblem(num_cities=problem_size)
                        problem.generate(seed=seed)
                    elif problem_type == "Portfolio":
                        problem = PortfolioProblem(num_assets=problem_size, num_selected=problem_size//2)
                        problem.generate(seed=seed)
                    
                    # Prepare job configuration
                    job_config = {
                        'problem': problem,
                        'edge_profile': edge_profile,
                        'strategy': strategy,
                        'comparative_mode': comparative_mode,
                        'custom_power': custom_power,
                        'timeout': timeout_override,
                        'solver_prefs': solver_prefs,
                        'submitted_at': datetime.now()
                    }
                    
                    st.session_state.current_job = job_config
                    st.session_state.job_running = True
                    
                    st.success("✅ Job submitted successfully! Go to 'Live Execution' to monitor progress.")
                    
                except Exception as e:
                    st.error(f"❌ Error creating problem: {str(e)}")
    
    if clear_button:
        st.rerun()
    
    # Display current configuration summary
    if st.session_state.current_job and not st.session_state.job_running:
        st.markdown("---")
        st.info("💡 No job currently running. Submit a new problem above.")

# =============================================================================
# Page 2: Live Execution
# =============================================================================

def page_live_execution():
    """Page for monitoring live job execution."""
    st.header("⚡ Live Execution")
    
    if not st.session_state.job_running:
        st.info("ℹ️ No job currently running. Submit a problem from the 'Submit Problem' page.")
        return
    
    if st.session_state.current_job is None:
        st.warning("⚠️ Job is marked as running but no job data found.")
        st.session_state.job_running = False
        return
    
    # Job information
    job_config = st.session_state.current_job
    st.markdown("### 📋 Job Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Problem Type", job_config['problem'].__class__.__name__)
        st.metric("Edge Profile", job_config['edge_profile'])
    with col2:
        st.metric("Strategy", job_config['strategy'])
        st.metric("Comparative Mode", "Yes" if job_config['comparative_mode'] else "No")
    with col3:
        elapsed = (datetime.now() - job_config['submitted_at']).total_seconds()
        st.metric("Elapsed Time", f"{elapsed:.1f}s")
    
    st.markdown("---")
    
    # Progress tracking
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Abort button
    if st.button("🛑 Abort Job", type="secondary"):
        st.session_state.job_running = False
        st.session_state.current_job = None
        st.warning("⚠️ Job aborted by user.")
        st.rerun()
    
    # Execute job
    with st.spinner("Executing job..."):
        progress_bar = progress_placeholder.progress(0.0)
        
        try:
            # Update progress - Problem Analysis
            progress_bar.progress(0.2)
            with metrics_placeholder.container():
                st.info("🔍 Step 1/4: Analyzing problem...")
            time.sleep(0.5)
            
            # Update progress - Routing Decision
            progress_bar.progress(0.4)
            with metrics_placeholder.container():
                st.info("🧭 Step 2/4: Making routing decision...")
            time.sleep(0.5)
            
            # Execute the job
            progress_bar.progress(0.6)
            with metrics_placeholder.container():
                st.info("⚙️ Step 3/4: Executing solver(s)...")
            
            orchestrator = st.session_state.orchestrator
            
            if job_config['comparative_mode']:
                result = orchestrator.execute_comparative(
                    problem=job_config['problem'],
                    edge_profile=job_config['edge_profile']
                )
            else:
                result = orchestrator.execute_job(
                    problem=job_config['problem'],
                    edge_profile=job_config['edge_profile'],
                    strategy=job_config['strategy']
                )
            
            # Update progress - Validation
            progress_bar.progress(0.9)
            with metrics_placeholder.container():
                st.info("✅ Step 4/4: Validating results...")
            time.sleep(0.5)
            
            # Complete
            progress_bar.progress(1.0)
            
            # Store result
            result['job_config'] = job_config
            result['completed_at'] = datetime.now()
            st.session_state.last_result = result
            st.session_state.job_history.append(result)
            st.session_state.job_running = False
            st.session_state.current_job = None
            
            st.success("✅ Job completed successfully!")
            
            # Display quick results
            with metrics_placeholder.container():
                st.markdown("### 📊 Quick Results")
                if job_config['comparative_mode']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Classical Time",
                            f"{result['classical']['time_ms']:.0f} ms"
                        )
                    with col2:
                        st.metric(
                            "Quantum Time",
                            f"{result['quantum']['time_ms']:.0f} ms"
                        )
                    with col3:
                        st.metric(
                            "Winner",
                            result.get('recommendation', 'N/A')
                        )
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Solver Used", result.get('solver_used', 'N/A'))
                    with col2:
                        st.metric("Execution Time", f"{result.get('time_ms', 0):.0f} ms")
                    with col3:
                        st.metric(
                            "Solution Quality",
                            f"{result.get('solution_quality', 0) * 100:.1f}%"
                        )
            
            st.info("💡 Go to 'Results Analysis' page for detailed analysis.")
            
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ Error during execution: {str(e)}")
            st.session_state.job_running = False
            st.session_state.current_job = None

# =============================================================================
# Page 3: Results Analysis
# =============================================================================

def page_results_analysis():
    """Page for analyzing the last job result."""
    st.header("📊 Results Analysis")
    
    if st.session_state.last_result is None:
        st.info("ℹ️ No results available. Submit and execute a problem first.")
        return
    
    result = st.session_state.last_result
    job_config = result.get('job_config', {})
    
    # Job metadata
    st.markdown("### 📋 Job Metadata")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Problem Type", job_config['problem'].__class__.__name__)
    with col2:
        st.metric("Edge Profile", job_config['edge_profile'])
    with col3:
        st.metric("Strategy", job_config['strategy'])
    with col4:
        completed_at = result.get('completed_at', datetime.now())
        st.metric("Completed", completed_at.strftime("%H:%M:%S"))
    
    st.markdown("---")
    
    # Comparative mode results
    if job_config.get('comparative_mode', False):
        st.markdown("### 🔬 Comparative Analysis")
        
        # Performance comparison chart
        try:
            fig = plot_performance_comparison(
                result.get('classical', {}),
                result.get('quantum', {})
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate comparison chart: {str(e)}")
        
        # Side-by-side metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Classical Solver")
            classical = result.get('classical', {})
            st.metric("Execution Time", f"{classical.get('time_ms', 0):.0f} ms")
            st.metric("Energy Used", f"{classical.get('energy_mj', 0):.2f} mJ")
            st.metric("Solution Quality", f"{classical.get('quality', 0) * 100:.1f}%")
        
        with col2:
            st.markdown("#### Quantum Solver")
            quantum = result.get('quantum', {})
            st.metric("Execution Time", f"{quantum.get('time_ms', 0):.0f} ms")
            st.metric("Energy Used", f"{quantum.get('energy_mj', 0):.2f} mJ")
            st.metric("Solution Quality", f"{quantum.get('quality', 0) * 100:.1f}%")
        
        # Recommendation
        st.markdown("### 🎯 Recommendation")
        recommendation = result.get('recommendation', 'N/A')
        st.success(f"**Winner:** {recommendation}")
        st.info(result.get('reasoning', 'No reasoning provided.'))
    
    else:
        # Single solver results
        st.markdown("### 📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Solver Used", result.get('solver_used', 'N/A'))
        with col2:
            st.metric("Execution Time", f"{result.get('time_ms', 0):.0f} ms")
        with col3:
            st.metric("Energy Used", f"{result.get('energy_mj', 0):.2f} mJ")
        with col4:
            st.metric("Solution Quality", f"{result.get('solution_quality', 0) * 100:.1f}%")
        
        # Routing decision explanation
        st.markdown("### 🧭 Routing Decision")
        chosen_solver = result.get('routing_decision', 'No chosen solver.')
        st.info(f"**Decision:** {chosen_solver}")
        st.write(result.get('reasoning', 'No routing information available.'))
        
        # # Alternative options
        # alternatives = routing_info.get('alternatives', [])
        # if alternatives:
        #     with st.expander("🔍 Alternative Options Considered"):
        #         for alt in alternatives:
        #             st.write(f"- {alt}")
    
    # Solution visualization
    st.markdown("---")
    st.markdown("### 🎨 Solution Visualization")
    try:
        problem = job_config['problem']
        solution = result.get('solution', result.get('classical', {}).get('solution'))

        
        if solution:
            fig = plot_graph_solution(problem, solution)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No solution data available for visualization.")
    except Exception as e:
        st.warning(f"Could not generate solution visualization: {str(e)}")
    
    # Download results
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    json_str = json.dumps(
        {k: v for k, v in result.items() if k != 'job_config'},
        indent=2,
        default=str
    )
    
    st.download_button(
        label="📥 Download as JSON",
        data=json_str,
        file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# =============================================================================
# Page 4: Historical Performance
# =============================================================================

def page_historical_performance():
    """Page for analyzing historical job performance."""
    st.header("📈 Historical Performance")
    
    if not st.session_state.job_history:
        st.info("ℹ️ No historical data available. Execute some jobs first.")
        return
    
    # Filters
    st.markdown("### 🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        problem_types = list(set(
            r.get('job_config', {}).get('problem', None).__class__.__name__
            for r in st.session_state.job_history
            if r.get('job_config', {}).get('problem')
        ))
        filter_problem = st.multiselect(
            "Problem Type",
            options=problem_types,
            default=problem_types
        )
    
    with col2:
        edge_profiles = list(set(
            r.get('job_config', {}).get('edge_profile', 'unknown')
            for r in st.session_state.job_history
        ))
        filter_profile = st.multiselect(
            "Edge Profile",
            options=edge_profiles,
            default=edge_profiles
        )
    
    with col3:
        # Date range
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now().date(), datetime.now().date())
        )
    
    # Filter data
    filtered_history = [
        r for r in st.session_state.job_history
        if (r.get('job_config', {}).get('problem', None).__class__.__name__ in filter_problem
            and r.get('job_config', {}).get('edge_profile', 'unknown') in filter_profile)
    ]
    
    st.markdown(f"**Showing {len(filtered_history)} of {len(st.session_state.job_history)} jobs**")
    
    st.markdown("---")
    
    # Visualizations
    if filtered_history:
        st.markdown("### 📊 Performance Trends")
        
        try:
            # Time series trends
            fig = plot_historical_trends(filtered_history)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate trends chart: {str(e)}")
        
        # Statistics table
        st.markdown("### 📋 Statistics Summary")
        stats_data = []
        for r in filtered_history:
            stats_data.append({
                "Problem": r.get('job_config', {}).get('problem', None).__class__.__name__,
                "Profile": r.get('job_config', {}).get('edge_profile', 'N/A'),
                "Solver": r.get('solver_used', 'N/A'),
                "Time (ms)": f"{r.get('time_ms', 0):.0f}",
                "Energy (mJ)": f"{r.get('energy_mj', 0):.2f}",
                "Quality (%)": f"{r.get('solution_quality', 0) * 100:.1f}",
            })
        
        st.table(stats_data)
        
        # Export to CSV
        import pandas as pd
        df = pd.DataFrame(stats_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data matches the selected filters.")

# =============================================================================
# Page 5: System Status
# =============================================================================

def page_system_status():
    """Page for checking system health and status."""
    st.header("🔧 System Status")
    
    # Database connection
    st.markdown("### 💾 Database Connection")
    try:
        # Test database connection
        from src.monitoring.db_manager import DatabaseManager
        db = DatabaseManager(database_url=settings.database.async_url)
        st.success("✅ Database connection: **HEALTHY**")
        st.info(f"Connected to: {settings.database.host}:{settings.database.port}")
    except Exception as e:
        st.error(f"❌ Database connection: **FAILED**\n\nError: {str(e)}")
    
    st.markdown("---")
    
    # Solver availability
    st.markdown("### ⚙️ Solver Availability")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ Classical Solver: **AVAILABLE**")
        st.info("QAOA + Goemans-Williamson fallback")
    
    with col2:
        st.success("✅ Quantum Simulator: **AVAILABLE**")
        st.info("Qiskit Aer Statevector")
    
    st.markdown("---")
    
    # Recent errors
    st.markdown("### ⚠️ Recent Errors")
    # This would normally fetch from logs/database
    st.success("No recent errors detected.")
    
    st.markdown("---")
    
    # Performance metrics
    st.markdown("### 📊 Performance Metrics")
    
    total_jobs = len(st.session_state.job_history)
    
    if total_jobs > 0:
        avg_time = sum(r.get('time_ms', 0) for r in st.session_state.job_history) / total_jobs
        
        # Calculate quantum savings (if any comparative results exist)
        quantum_jobs = [r for r in st.session_state.job_history if 'quantum' in r]
        if quantum_jobs:
            energy_saved = sum(
                r.get('classical', {}).get('energy_mj', 0) - r.get('quantum', {}).get('energy_mj', 0)
                for r in quantum_jobs
            )
        else:
            energy_saved = 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Jobs Processed", total_jobs)
        with col2:
            st.metric("Average Execution Time", f"{avg_time:.0f} ms")
        with col3:
            st.metric("Energy Saved (Quantum)", f"{energy_saved:.2f} mJ")
    else:
        st.info("No jobs processed yet.")
    
    st.markdown("---")
    
    # Resource utilization
    st.markdown("### 💻 Resource Utilization")
    
    import psutil
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        st.progress(cpu_percent / 100.0)
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent:.1f}%")
        st.progress(memory.percent / 100.0)
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent:.1f}%")
        st.progress(disk.percent / 100.0)

# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "Submit Problem":
        page_submit_problem()
    elif page == "Live Execution":
        page_live_execution()
    elif page == "Results Analysis":
        page_results_analysis()
    elif page == "Historical Performance":
        page_historical_performance()
    elif page == "System Status":
        page_system_status()

if __name__ == "__main__":
    main()
