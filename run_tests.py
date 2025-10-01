#!/usr/bin/env python3
"""
Test runner script for astron-result test suite.
Provides convenient commands to run different test categories.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def run_all_tests():
    """Run all tests with verbose output."""
    return run_command([
        "uv", "run", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short",
        "--durations=10"
    ])


def run_unit_tests():
    """Run unit tests only."""
    return run_command([
        "uv", "run", "pytest", 
        "tests/test_astron_result.py", 
        "-v", 
        "--tb=short"
    ])


def run_integration_tests():
    """Run integration tests only."""
    return run_command([
        "uv", "run", "pytest", 
        "tests/test_integration.py", 
        "-v", 
        "--tb=short"
    ])


def run_with_coverage():
    """Run tests with coverage report."""
    return run_command([
        "uv", "run", "pytest", 
        "tests/", 
        "--cov=astron",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ])


def run_performance_tests():
    """Run performance-related tests only."""
    return run_command([
        "uv", "run", "pytest", 
        "tests/test_astron_result.py::TestPerformanceAndMemory",
        "-v", 
        "--tb=short"
    ])


def main():
    """Main test runner function."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <command>")
        print("\nAvailable commands:")
        print("  all        - Run all tests")
        print("  unit       - Run unit tests only")
        print("  integration - Run integration tests only")
        print("  coverage   - Run tests with coverage report")
        print("  performance - Run performance tests only")
        print("  help       - Show this help message")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "all":
        return run_all_tests()
    elif command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "coverage":
        return run_with_coverage()
    elif command == "performance":
        return run_performance_tests()
    elif command == "help":
        main()
        return 0
    else:
        print(f"Unknown command: {command}")
        print("Run 'python run_tests.py help' for available commands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())