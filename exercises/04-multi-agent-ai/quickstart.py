#!/usr/bin/env python3
"""
Quick start script for multi-agent system
Demonstrates basic usage and example workflows
"""

import requests
import json
import time


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def execute_task(description, constraints=None):
    """Execute a task via API"""
    url = "http://localhost:5000/task"
    
    payload = {"description": description}
    if constraints:
        payload["constraints"] = constraints
    
    print(f"📝 Task: {description}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Status: {result['status']}")
        print(f"🆔 Task ID: {result['task_id']}")
        print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
        print(f"🤖 Agents Used: {', '.join(result['agents_used'])}")
        
        if 'evaluation' in result:
            eval_data = result['evaluation']
            print(f"📊 Evaluation Score: {eval_data['evaluation']['score']}")
            print(f"✓  Passes Threshold: {eval_data['passes_threshold']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {str(e)}")
        return None


def check_health():
    """Check API health"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ System Status: {data['status']}")
        print(f"📦 Version: {data['version']}")
        return True
    except requests.exceptions.RequestException:
        print("❌ API is not responding. Please start the server first.")
        print("\nStart with: python src/api.py")
        print("Or with Docker: docker-compose up")
        return False


def get_metrics():
    """Get system metrics"""
    try:
        response = requests.get("http://localhost:5000/metrics", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        system_metrics = data['system_metrics']
        success_criteria = data['success_criteria']
        
        print(f"📈 Total Tasks: {system_metrics['total_tasks']}")
        print(f"✓  Completion Rate: {system_metrics['avg_completion_rate']:.2%}")
        print(f"🤝 Coordination Efficiency: {system_metrics['avg_coordination_efficiency']:.2f}")
        print(f"⏱️  Avg Execution Time: {system_metrics['avg_execution_time']:.2f}s")
        
        print("\n🎯 Success Criteria:")
        print(f"   Overall: {'✅ PASS' if success_criteria['overall_pass'] else '❌ FAIL'}")
        print(f"   Completion: {success_criteria['completion_rate']['value']:.2%} "
              f"(threshold: {success_criteria['completion_rate']['threshold']:.0%})")
        print(f"   Efficiency: {success_criteria['coordination_efficiency']['value']:.2f} "
              f"(threshold: {success_criteria['coordination_efficiency']['threshold']})")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching metrics: {str(e)}")


def main():
    """Run quick start examples"""
    print("\n" + "=" * 60)
    print("  Multi-Agent System - Quick Start")
    print("=" * 60)
    
    # Check health
    print_section("1. Health Check")
    if not check_health():
        return
    
    print("\nWaiting 2 seconds before starting examples...")
    time.sleep(2)
    
    # Example 1: Simple task
    print_section("2. Example: Simple Task")
    execute_task("Research machine learning best practices")
    
    time.sleep(1)
    
    # Example 2: Task with constraints
    print_section("3. Example: Task with Constraints")
    execute_task(
        "Analyze and optimize database query performance",
        constraints={"quality_threshold": 0.8}
    )
    
    time.sleep(1)
    
    # Example 3: Complex research task
    print_section("4. Example: Complex Research Task")
    execute_task("Research, evaluate, and implement API rate limiting strategies")
    
    time.sleep(1)
    
    # Get system metrics
    print_section("5. System Metrics")
    get_metrics()
    
    # Summary
    print_section("Summary")
    print("✅ Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Explore the API: http://localhost:5000")
    print("2. View metrics: http://localhost:9090 (Prometheus)")
    print("3. Check documentation: README.md")
    print("4. Run tests: make test")
    print()


if __name__ == "__main__":
    main()
