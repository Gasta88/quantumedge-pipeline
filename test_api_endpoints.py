"""
Simple test script to verify API endpoints are working.
"""
import sys
import asyncio
from fastapi.testclient import TestClient

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    print("Testing /health endpoint...", end=" ")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    print("✓ PASSED")
    return True

def test_system_info():
    """Test system info endpoint."""
    print("Testing /api/v1/system/info endpoint...", end=" ")
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "application" in data
    assert "configuration" in data
    assert "capabilities" in data
    print("✓ PASSED")
    return True

def test_edge_profiles():
    """Test edge profiles endpoint."""
    print("Testing /api/v1/config/edge-profiles endpoint...", end=" ")
    response = client.get("/api/v1/config/edge-profiles")
    assert response.status_code == 200
    data = response.json()
    assert "profiles" in data
    assert "aerospace" in data["profiles"]
    assert "mobile" in data["profiles"]
    assert "ground" in data["profiles"]
    print("✓ PASSED")
    return True

def test_routing_strategies():
    """Test routing strategies endpoint."""
    print("Testing /api/v1/config/routing-strategies endpoint...", end=" ")
    response = client.get("/api/v1/config/routing-strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert "balanced" in data["strategies"]
    print("✓ PASSED")
    return True

def test_routing_analysis():
    """Test routing analysis endpoint."""
    print("Testing /api/v1/routing/analyze endpoint...", end=" ")
    request_data = {
        "problem_type": "maxcut",
        "problem_size": 20,
        "edge_profile": "aerospace",
        "strategy": "balanced"
    }
    response = client.post("/api/v1/routing/analyze", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "reasoning" in data
    assert "confidence" in data
    print("✓ PASSED")
    return True

def test_maxcut_job_submission():
    """Test MaxCut job submission endpoint."""
    print("Testing /api/v1/jobs/maxcut endpoint...", end=" ")
    request_data = {
        "num_nodes": 15,
        "edge_probability": 0.3,
        "edge_profile": "aerospace",
        "strategy": "balanced",
        "seed": 42
    }
    response = client.post("/api/v1/jobs/maxcut", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ["completed", "failed"]
    assert data["problem_type"] == "maxcut"
    assert data["problem_size"] == 15
    print("✓ PASSED")
    print(f"    Job ID: {data['job_id']}")
    print(f"    Solver: {data.get('solver_used', 'N/A')}")
    print(f"    Time: {data.get('time_ms', 0):.2f} ms")
    return True

def test_comparative_job_submission():
    """Test comparative job submission endpoint."""
    print("Testing /api/v1/jobs/comparative endpoint...", end=" ")
    request_data = {
        "problem_type": "maxcut",
        "problem_size": 10,
        "edge_profile": "aerospace",
        "seed": 42
    }
    response = client.post("/api/v1/jobs/comparative", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "success" in data
    assert "classical" in data
    assert "quantum" in data
    assert "recommendation" in data
    print("✓ PASSED")
    print(f"    Job ID: {data['job_id']}")
    print(f"    Recommendation: {data['recommendation']}")
    if data.get('speedup_factor'):
        print(f"    Speedup: {data['speedup_factor']:.2f}x")
    return True

def main():
    """Run all tests."""
    print("=" * 80)
    print("QUANTUMEDGE PIPELINE API - ENDPOINT TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_health_check,
        test_system_info,
        test_edge_profiles,
        test_routing_strategies,
        test_routing_analysis,
        test_maxcut_job_submission,
        test_comparative_job_submission,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
