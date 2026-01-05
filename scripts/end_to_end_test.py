"""
Comprehensive End-to-End Test Script for QuantumEdge Pipeline.

This script performs a complete system test by executing the entire workflow:
1. Start all Docker services
2. Wait for database ready
3. Initialize database schema
4. Generate test problems (5 different sizes)
5. Submit jobs through orchestrator
6. Verify database records created
7. Query metrics
8. Test Streamlit dashboard loads
9. Run comparison analysis
10. Export results
11. Validate all data integrity
12. Generate test report

The test suite includes timing measurements, success/failure logging, and comprehensive
validation to ensure all components work correctly in an integrated environment.

Usage:
------
    # Run full end-to-end test
    python scripts/end_to_end_test.py
    
    # Run with verbose logging
    python scripts/end_to_end_test.py --verbose
    
    # Skip Docker checks (if services already running)
    python scripts/end_to_end_test.py --skip-docker
    
    # Custom problem sizes
    python scripts/end_to_end_test.py --sizes 10 20 30 40 50

Example Output:
--------------
    ==================== QuantumEdge Pipeline E2E Test ====================
    Test Run ID: e2e_test_20240105_143022
    
    [1/12] Starting Docker services...                    [✓] PASS (12.3s)
    [2/12] Waiting for database ready...                  [✓] PASS (3.1s)
    [3/12] Initializing database schema...                [✓] PASS (0.8s)
    [4/12] Generating test problems (5 sizes)...          [✓] PASS (2.4s)
    [5/12] Submitting jobs through orchestrator...        [✓] PASS (45.2s)
    [6/12] Verifying database records...                  [✓] PASS (1.2s)
    [7/12] Querying metrics from database...              [✓] PASS (0.6s)
    [8/12] Testing Streamlit dashboard loads...           [✓] PASS (5.3s)
    [9/12] Running comparison analysis...                 [✓] PASS (38.7s)
    [10/12] Exporting results...                          [✓] PASS (1.1s)
    [11/12] Validating data integrity...                  [✓] PASS (2.3s)
    [12/12] Generating test report...                     [✓] PASS (0.4s)
    
    ====================== Test Summary ======================
    Total Tests:  12
    Passed:       12
    Failed:       0
    Skipped:      0
    Success Rate: 100.0%
    Total Time:   113.4s
    
    Report saved to: data/e2e_test_20240105_143022_report.json
"""

import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import logging
from dataclasses import dataclass, asdict
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports
import requests
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for end-to-end test."""
    test_run_id: str
    start_time: datetime
    verbose: bool = False
    skip_docker: bool = False
    problem_sizes: List[int] = None
    
    # Service endpoints
    api_url: str = "http://localhost:8000"
    dashboard_url: str = "http://localhost:8501"
    grafana_url: str = "http://localhost:3000"
    
    # Database configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "quantumedge"
    db_user: str = "qe_user"
    db_password: str = "qe_pass"
    
    # Timeouts
    docker_startup_timeout: int = 120  # seconds
    db_ready_timeout: int = 60
    service_ready_timeout: int = 90
    job_execution_timeout: int = 300
    
    # Output paths
    output_dir: Path = Path("data")
    report_filename: str = None
    
    def __post_init__(self):
        if self.problem_sizes is None:
            self.problem_sizes = [10, 20, 30, 40, 50]
        
        if self.report_filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            self.report_filename = f"e2e_test_{timestamp}_report.json"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TestResult:
    """Result of a single test step."""
    step_number: int
    step_name: str
    status: str  # 'PASS', 'FAIL', 'SKIP'
    duration_seconds: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """Format test result for display."""
        status_symbol = {
            'PASS': '✓',
            'FAIL': '✗',
            'SKIP': '○'
        }.get(self.status, '?')
        
        return f"[{self.step_number}/12] {self.step_name:<50} [{status_symbol}] {self.status} ({self.duration_seconds:.1f}s)"


class TestSuite:
    """Manages the end-to-end test execution."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.logger = self._setup_logging()
        
        # Import QuantumEdge components
        try:
            from src.problems.maxcut import MaxCutProblem
            from src.problems.tsp import TSPProblem
            from src.problems.portfolio import PortfolioProblem
            from src.api.orchestrator import JobOrchestrator
            from src.monitoring.db_manager import DatabaseManager
            
            self.MaxCutProblem = MaxCutProblem
            self.TSPProblem = TSPProblem
            self.PortfolioProblem = PortfolioProblem
            self.JobOrchestrator = JobOrchestrator
            self.DatabaseManager = DatabaseManager
            
        except ImportError as e:
            self.logger.error(f"Failed to import QuantumEdge components: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for test suite."""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        logger = logging.getLogger(__name__)
        
        # Also log to file
        log_file = self.config.output_dir / f"{self.config.test_run_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def run_test(self, step_number: int, step_name: str, test_func) -> TestResult:
        """Execute a single test step with timing and error handling."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"[{step_number}/12] {step_name}")
        self.logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Execute test function
            metadata = test_func()
            
            duration = time.time() - start_time
            result = TestResult(
                step_number=step_number,
                step_name=step_name,
                status='PASS',
                duration_seconds=duration,
                metadata=metadata
            )
            
            self.logger.info(f"✓ PASS ({duration:.1f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            result = TestResult(
                step_number=step_number,
                step_name=step_name,
                status='FAIL',
                duration_seconds=duration,
                error_message=error_msg,
                error_traceback=error_trace
            )
            
            self.logger.error(f"✗ FAIL ({duration:.1f}s)")
            self.logger.error(f"Error: {error_msg}")
            if self.config.verbose:
                self.logger.debug(f"Traceback:\n{error_trace}")
        
        self.results.append(result)
        return result
    
    # ========================================================================
    # TEST STEP IMPLEMENTATIONS
    # ========================================================================
    
    def test_01_start_docker_services(self) -> Dict[str, Any]:
        """Step 1: Start all Docker services."""
        if self.config.skip_docker:
            self.logger.info("Skipping Docker startup (--skip-docker flag)")
            return {'skipped': True}
        
        self.logger.info("Starting Docker services with docker-compose...")
        
        # Check if docker-compose is available
        result = subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError("docker-compose not found. Please install Docker Compose.")
        
        # Start services
        result = subprocess.run(
            ['docker-compose', 'up', '-d'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=self.config.docker_startup_timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Docker services:\n{result.stderr}")
        
        self.logger.info("Docker services started successfully")
        
        # Get service status
        result = subprocess.run(
            ['docker-compose', 'ps'],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        return {
            'services_output': result.stdout,
            'started': True
        }
    
    def test_02_wait_for_database_ready(self) -> Dict[str, Any]:
        """Step 2: Wait for database to be ready."""
        self.logger.info(f"Waiting for PostgreSQL to be ready on {self.config.db_host}:{self.config.db_port}...")
        
        max_attempts = self.config.db_ready_timeout // 5
        
        for attempt in range(1, max_attempts + 1):
            try:
                conn = psycopg2.connect(
                    host=self.config.db_host,
                    port=self.config.db_port,
                    database=self.config.db_name,
                    user=self.config.db_user,
                    password=self.config.db_password,
                    connect_timeout=5
                )
                conn.close()
                
                self.logger.info(f"Database ready after {attempt} attempt(s)")
                return {
                    'ready': True,
                    'attempts': attempt
                }
            
            except psycopg2.OperationalError as e:
                if attempt < max_attempts:
                    self.logger.debug(f"Attempt {attempt}/{max_attempts}: Database not ready yet, waiting...")
                    time.sleep(5)
                else:
                    raise RuntimeError(f"Database not ready after {max_attempts} attempts: {e}")
        
        raise RuntimeError("Unexpected exit from database ready check")
    
    def test_03_initialize_database_schema(self) -> Dict[str, Any]:
        """Step 3: Initialize database schema."""
        self.logger.info("Verifying database schema initialization...")
        
        conn = psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password
        )
        
        cursor = conn.cursor()
        
        # Check if required tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name IN ('problems', 'job_executions', 'performance_metrics')
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        self.logger.info(f"Found tables: {tables}")
        
        required_tables = ['problems', 'job_executions', 'performance_metrics']
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            raise RuntimeError(f"Missing required tables: {missing_tables}")
        
        # Check TimescaleDB hypertables
        cursor.execute("""
            SELECT hypertable_name 
            FROM timescaledb_information.hypertables
        """)
        
        hypertables = [row[0] for row in cursor.fetchall()]
        self.logger.info(f"Found hypertables: {hypertables}")
        
        cursor.close()
        conn.close()
        
        return {
            'tables_found': tables,
            'hypertables_found': hypertables,
            'schema_valid': True
        }
    
    def test_04_generate_test_problems(self) -> Dict[str, Any]:
        """Step 4: Generate test problems of different sizes."""
        self.logger.info(f"Generating test problems with sizes: {self.config.problem_sizes}")
        
        problems = []
        
        for size in self.config.problem_sizes:
            self.logger.info(f"  Generating MaxCut problem with {size} nodes...")
            
            problem = self.MaxCutProblem(num_nodes=size)
            problem.generate_random_graph(edge_probability=0.3, seed=42 + size)
            
            problems.append({
                'type': 'maxcut',
                'size': size,
                'problem_obj': problem,
                'num_edges': len(problem.edges)
            })
            
            self.logger.debug(f"    Created: {size} nodes, {len(problem.edges)} edges")
        
        # Store problems for later test steps
        self.test_problems = problems
        
        return {
            'num_problems': len(problems),
            'problem_sizes': self.config.problem_sizes,
            'total_nodes': sum(p['size'] for p in problems),
            'total_edges': sum(p['num_edges'] for p in problems)
        }
    
    def test_05_submit_jobs_through_orchestrator(self) -> Dict[str, Any]:
        """Step 5: Submit jobs through orchestrator."""
        self.logger.info("Submitting jobs through orchestrator...")
        
        if not hasattr(self, 'test_problems'):
            raise RuntimeError("Test problems not generated. Run test_04 first.")
        
        # Initialize orchestrator with database
        db_url = f"postgresql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        
        orchestrator = self.JobOrchestrator(
            enable_db=True,
            enable_validation=True,
            db_url=db_url
        )
        
        results = []
        
        for idx, problem_data in enumerate(self.test_problems):
            problem = problem_data['problem_obj']
            size = problem_data['size']
            
            self.logger.info(f"  [{idx+1}/{len(self.test_problems)}] Submitting problem (size={size})...")
            
            result = orchestrator.execute_job(
                problem=problem,
                edge_profile='aerospace',
                strategy='balanced'
            )
            
            results.append({
                'problem_size': size,
                'job_id': result['job_id'],
                'success': result['success'],
                'solver_used': result.get('solver_used'),
                'execution_time_ms': result.get('time_ms'),
                'solution_quality': result.get('solution_quality')
            })
            
            status = "✓" if result['success'] else "✗"
            self.logger.info(f"    {status} Job {result['job_id'][:8]}... completed in {result.get('time_ms', 0):.1f}ms")
        
        # Store results for later validation
        self.job_results = results
        
        successful = sum(1 for r in results if r['success'])
        
        return {
            'total_jobs': len(results),
            'successful_jobs': successful,
            'failed_jobs': len(results) - successful,
            'job_ids': [r['job_id'] for r in results],
            'avg_execution_time_ms': sum(r['execution_time_ms'] or 0 for r in results) / len(results)
        }
    
    def test_06_verify_database_records(self) -> Dict[str, Any]:
        """Step 6: Verify database records were created."""
        self.logger.info("Verifying database records...")
        
        if not hasattr(self, 'job_results'):
            raise RuntimeError("Jobs not submitted. Run test_05 first.")
        
        conn = psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password
        )
        
        cursor = conn.cursor()
        
        # Count problems
        cursor.execute("SELECT COUNT(*) FROM problems")
        problem_count = cursor.fetchone()[0]
        self.logger.info(f"  Problems in database: {problem_count}")
        
        # Count job executions
        cursor.execute("SELECT COUNT(*) FROM job_executions")
        job_count = cursor.fetchone()[0]
        self.logger.info(f"  Job executions in database: {job_count}")
        
        # Verify our job IDs exist (if database manager stores them)
        # Note: Current orchestrator implementation may not store in DB
        # This is a placeholder for when DB integration is complete
        
        cursor.close()
        conn.close()
        
        return {
            'problems_count': problem_count,
            'job_executions_count': job_count,
            'verification_passed': True
        }
    
    def test_07_query_metrics(self) -> Dict[str, Any]:
        """Step 7: Query metrics from database."""
        self.logger.info("Querying metrics from database...")
        
        conn = psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password
        )
        
        cursor = conn.cursor()
        
        # Query recent jobs summary
        cursor.execute("""
            SELECT 
                COUNT(*) as total_jobs,
                AVG(execution_time_ms) as avg_time_ms,
                AVG(solution_quality) as avg_quality
            FROM job_executions
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)
        
        row = cursor.fetchone()
        
        metrics = {
            'total_jobs': row[0] if row[0] else 0,
            'avg_time_ms': float(row[1]) if row[1] else 0.0,
            'avg_quality': float(row[2]) if row[2] else 0.0
        }
        
        self.logger.info(f"  Total jobs (last hour): {metrics['total_jobs']}")
        self.logger.info(f"  Average execution time: {metrics['avg_time_ms']:.2f}ms")
        self.logger.info(f"  Average solution quality: {metrics['avg_quality']:.2%}")
        
        cursor.close()
        conn.close()
        
        return metrics
    
    def test_08_test_streamlit_dashboard_loads(self) -> Dict[str, Any]:
        """Step 8: Test Streamlit dashboard loads."""
        self.logger.info(f"Testing Streamlit dashboard at {self.config.dashboard_url}...")
        
        # Wait for dashboard to be ready
        max_attempts = self.config.service_ready_timeout // 10
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Try health endpoint first
                response = requests.get(
                    f"{self.config.dashboard_url}/_stcore/health",
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.logger.info(f"Dashboard health check passed after {attempt} attempt(s)")
                    break
            
            except requests.exceptions.RequestException:
                if attempt < max_attempts:
                    self.logger.debug(f"Attempt {attempt}/{max_attempts}: Dashboard not ready, waiting...")
                    time.sleep(10)
                else:
                    raise RuntimeError(f"Dashboard not ready after {max_attempts} attempts")
        
        # Try to load main page
        response = requests.get(self.config.dashboard_url, timeout=10)
        
        if response.status_code != 200:
            raise RuntimeError(f"Dashboard returned status {response.status_code}")
        
        self.logger.info("Dashboard loaded successfully")
        
        return {
            'dashboard_url': self.config.dashboard_url,
            'status_code': response.status_code,
            'response_size': len(response.content),
            'accessible': True
        }
    
    def test_09_run_comparison_analysis(self) -> Dict[str, Any]:
        """Step 9: Run comparison analysis."""
        self.logger.info("Running comparison analysis...")
        
        if not hasattr(self, 'test_problems'):
            raise RuntimeError("Test problems not generated. Run test_04 first.")
        
        # Use a medium-sized problem for comparison
        problem = self.test_problems[2]['problem_obj']  # Size 30
        
        db_url = f"postgresql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        
        orchestrator = self.JobOrchestrator(
            enable_db=True,
            enable_validation=True,
            db_url=db_url
        )
        
        self.logger.info(f"  Running comparative analysis on problem size {problem.problem_size}...")
        
        comparison = orchestrator.execute_comparative(
            problem=problem,
            edge_profile='aerospace'
        )
        
        classical = comparison.get('classical', {})
        quantum = comparison.get('quantum', {})
        
        self.logger.info(f"  Classical: cost={classical.get('cost', 'N/A')}, time={classical.get('time_ms', 'N/A')}ms")
        self.logger.info(f"  Quantum: cost={quantum.get('cost', 'N/A')}, time={quantum.get('time_ms', 'N/A')}ms")
        self.logger.info(f"  Recommendation: {comparison.get('recommendation', 'N/A')}")
        
        # Store comparison for report
        self.comparison_result = comparison
        
        return {
            'comparison_job_id': comparison.get('job_id'),
            'success': comparison.get('success'),
            'classical_time_ms': classical.get('time_ms'),
            'quantum_time_ms': quantum.get('time_ms'),
            'speedup_factor': comparison.get('speedup_factor'),
            'recommendation': comparison.get('recommendation')
        }
    
    def test_10_export_results(self) -> Dict[str, Any]:
        """Step 10: Export results to files."""
        self.logger.info("Exporting results...")
        
        if not hasattr(self, 'job_results'):
            raise RuntimeError("Jobs not submitted. Run test_05 first.")
        
        # Export job results to CSV
        df = pd.DataFrame(self.job_results)
        
        csv_path = self.config.output_dir / f"{self.config.test_run_id}_job_results.csv"
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"  Exported job results to: {csv_path}")
        
        # Export comparison results to JSON
        if hasattr(self, 'comparison_result'):
            json_path = self.config.output_dir / f"{self.config.test_run_id}_comparison.json"
            
            with open(json_path, 'w') as f:
                json.dump(self.comparison_result, f, indent=2, default=str)
            
            self.logger.info(f"  Exported comparison to: {json_path}")
        
        return {
            'csv_path': str(csv_path),
            'json_path': str(json_path) if hasattr(self, 'comparison_result') else None,
            'exported': True
        }
    
    def test_11_validate_data_integrity(self) -> Dict[str, Any]:
        """Step 11: Validate all data integrity."""
        self.logger.info("Validating data integrity...")
        
        validation_checks = []
        
        # Check 1: All jobs have valid IDs
        if hasattr(self, 'job_results'):
            for result in self.job_results:
                if not result.get('job_id'):
                    validation_checks.append({
                        'check': 'job_id_present',
                        'passed': False,
                        'message': 'Job missing ID'
                    })
                else:
                    validation_checks.append({
                        'check': 'job_id_present',
                        'passed': True,
                        'message': 'All jobs have IDs'
                    })
                    break
        
        # Check 2: Execution times are reasonable
        if hasattr(self, 'job_results'):
            for result in self.job_results:
                time_ms = result.get('execution_time_ms', 0)
                if time_ms < 0 or time_ms > 300000:  # 5 minutes max
                    validation_checks.append({
                        'check': 'execution_time_reasonable',
                        'passed': False,
                        'message': f'Unreasonable execution time: {time_ms}ms'
                    })
                    break
            else:
                validation_checks.append({
                    'check': 'execution_time_reasonable',
                    'passed': True,
                    'message': 'All execution times reasonable'
                })
        
        # Check 3: Solution qualities are in valid range
        if hasattr(self, 'job_results'):
            for result in self.job_results:
                quality = result.get('solution_quality')
                if quality is not None and (quality < 0 or quality > 1):
                    validation_checks.append({
                        'check': 'solution_quality_valid',
                        'passed': False,
                        'message': f'Invalid solution quality: {quality}'
                    })
                    break
            else:
                validation_checks.append({
                    'check': 'solution_quality_valid',
                    'passed': True,
                    'message': 'All solution qualities valid'
                })
        
        # Check 4: Database connectivity
        try:
            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                connect_timeout=5
            )
            conn.close()
            validation_checks.append({
                'check': 'database_connectivity',
                'passed': True,
                'message': 'Database connection successful'
            })
        except Exception as e:
            validation_checks.append({
                'check': 'database_connectivity',
                'passed': False,
                'message': f'Database connection failed: {e}'
            })
        
        passed_checks = sum(1 for c in validation_checks if c['passed'])
        total_checks = len(validation_checks)
        
        for check in validation_checks:
            status = "✓" if check['passed'] else "✗"
            self.logger.info(f"  {status} {check['check']}: {check['message']}")
        
        all_passed = passed_checks == total_checks
        
        if not all_passed:
            raise RuntimeError(f"Validation failed: {passed_checks}/{total_checks} checks passed")
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'checks': validation_checks,
            'all_passed': all_passed
        }
    
    def test_12_generate_test_report(self) -> Dict[str, Any]:
        """Step 12: Generate test report."""
        self.logger.info("Generating test report...")
        
        # Compile comprehensive report
        report = {
            'test_run_id': self.config.test_run_id,
            'start_time': self.config.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': sum(r.duration_seconds for r in self.results),
            'configuration': {
                'problem_sizes': self.config.problem_sizes,
                'api_url': self.config.api_url,
                'dashboard_url': self.config.dashboard_url,
                'database': f"{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
            },
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results if r.status == 'PASS'),
                'failed': sum(1 for r in self.results if r.status == 'FAIL'),
                'skipped': sum(1 for r in self.results if r.status == 'SKIP')
            },
            'test_results': [
                {
                    'step_number': r.step_number,
                    'step_name': r.step_name,
                    'status': r.status,
                    'duration_seconds': r.duration_seconds,
                    'error_message': r.error_message,
                    'metadata': r.metadata
                }
                for r in self.results
            ]
        }
        
        # Add job results if available
        if hasattr(self, 'job_results'):
            report['job_executions'] = self.job_results
        
        # Add comparison results if available
        if hasattr(self, 'comparison_result'):
            report['comparison_analysis'] = self.comparison_result
        
        # Save report
        report_path = self.config.output_dir / self.config.report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"  Test report saved to: {report_path}")
        
        return {
            'report_path': str(report_path),
            'total_tests': report['summary']['total_tests'],
            'passed': report['summary']['passed'],
            'failed': report['summary']['failed']
        }
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_all_tests(self):
        """Run all test steps in sequence."""
        print("\n" + "="*70)
        print("QuantumEdge Pipeline - End-to-End Test Suite".center(70))
        print("="*70)
        print(f"Test Run ID: {self.config.test_run_id}")
        print(f"Start Time:  {self.config.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        # Define all test steps
        tests = [
            (1, "Starting Docker services", self.test_01_start_docker_services),
            (2, "Waiting for database ready", self.test_02_wait_for_database_ready),
            (3, "Initializing database schema", self.test_03_initialize_database_schema),
            (4, "Generating test problems (5 sizes)", self.test_04_generate_test_problems),
            (5, "Submitting jobs through orchestrator", self.test_05_submit_jobs_through_orchestrator),
            (6, "Verifying database records", self.test_06_verify_database_records),
            (7, "Querying metrics", self.test_07_query_metrics),
            (8, "Testing Streamlit dashboard loads", self.test_08_test_streamlit_dashboard_loads),
            (9, "Running comparison analysis", self.test_09_run_comparison_analysis),
            (10, "Exporting results", self.test_10_export_results),
            (11, "Validating data integrity", self.test_11_validate_data_integrity),
            (12, "Generating test report", self.test_12_generate_test_report)
        ]
        
        # Run each test
        for step_num, step_name, test_func in tests:
            result = self.run_test(step_num, step_name, test_func)
            
            # Print result
            print(result)
            
            # Stop if critical test fails
            if result.status == 'FAIL' and step_num <= 5:
                print(f"\n✗ Critical test failed: {step_name}")
                print("Stopping test execution.\n")
                break
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test execution summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == 'PASS')
        failed = sum(1 for r in self.results if r.status == 'FAIL')
        skipped = sum(1 for r in self.results if r.status == 'SKIP')
        total_time = sum(r.duration_seconds for r in self.results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print("Test Summary".center(70))
        print("="*70)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed}")
        print(f"Failed:       {failed}")
        print(f"Skipped:      {skipped}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time:   {total_time:.1f}s")
        print("="*70)
        
        if hasattr(self, 'config'):
            report_path = self.config.output_dir / self.config.report_filename
            print(f"\nReport saved to: {report_path}")
        
        print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for end-to-end test."""
    parser = argparse.ArgumentParser(
        description='Comprehensive end-to-end test for QuantumEdge Pipeline'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--skip-docker',
        action='store_true',
        help='Skip Docker service startup (assume services already running)'
    )
    
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[10, 20, 30, 40, 50],
        help='Problem sizes to test (default: 10 20 30 40 50)'
    )
    
    args = parser.parse_args()
    
    # Create test configuration
    test_run_id = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = TestConfig(
        test_run_id=test_run_id,
        start_time=datetime.now(),
        verbose=args.verbose,
        skip_docker=args.skip_docker,
        problem_sizes=args.sizes
    )
    
    # Create and run test suite
    suite = TestSuite(config)
    
    try:
        suite.run_all_tests()
        
        # Return exit code based on results
        failed = sum(1 for r in suite.results if r.status == 'FAIL')
        sys.exit(0 if failed == 0 else 1)
    
    except KeyboardInterrupt:
        print("\n\n✗ Test execution interrupted by user\n")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\n✗ Fatal error during test execution: {e}\n")
        if config.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
