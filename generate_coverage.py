#!/usr/bin/env python3
"""Generate combined coverage report for all services."""
import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_coverage(service_name: str) -> bool:
    """Run pytest with coverage for a service."""
    print(f"\n{'='*60}")
    print(f"Running coverage for {service_name}...")
    print(f"{'='*60}")
    
    service_dir = Path(__file__).parent / service_name
    if not service_dir.exists():
        print(f"Error: {service_name} directory not found")
        return False
    
    os.chdir(service_dir)
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "--cov=app",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=json",
                "-v"
            ],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running coverage for {service_name}: {e}")
        return False
    finally:
        os.chdir(Path(__file__).parent)


def main():
    """Generate coverage reports for all services."""
    root_dir = Path(__file__).parent
    coverage_dir = root_dir / "coverage"
    coverage_dir.mkdir(exist_ok=True)
    
    services = ["order-service", "menu-service", "logistics-service"]
    results = {}
    
    for service in services:
        success = run_coverage(service)
        results[service] = success
        
        # Copy HTML coverage report
        service_dir = root_dir / service
        htmlcov_dir = service_dir / "htmlcov"
        if htmlcov_dir.exists():
            target_dir = coverage_dir / f"{service}-htmlcov"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(htmlcov_dir, target_dir)
            print(f"\n✓ HTML coverage report copied to coverage/{service}-htmlcov/")
    
    # Summary
    print(f"\n{'='*60}")
    print("COVERAGE REPORT SUMMARY")
    print(f"{'='*60}")
    
    for service, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{service:25} {status}")
    
    print(f"\n{'='*60}")
    print("HTML Coverage Reports:")
    print(f"{'='*60}")
    for service in services:
        html_path = coverage_dir / f"{service}-htmlcov" / "index.html"
        if html_path.exists():
            print(f"  - {service}: coverage/{service}-htmlcov/index.html")
    
    print(f"\n{'='*60}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

