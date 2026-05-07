#!/usr/bin/env python3
"""
Generate all animations for Chapter 11 — Advanced Agentic Patterns

Runs all animation generator scripts in sequence and reports status.

Usage:
    python generate_all.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run all animation generators"""
    
    scripts = [
        "gen_ch11_reflection_loop.py",
        "gen_ch11_reflection_convergence.py",
        "gen_ch11_debate_flow.py",
        "gen_ch11_debate_consensus.py",
        "gen_ch11_hierarchical_flow.py",
        "gen_ch11_hierarchical_coordination.py",
        "gen_ch11_tool_selection_decision_tree.py",
        "gen_ch11_tool_fallback_chain.py",
        "gen_ch11_pattern_comparison.py",
        "gen_ch11_pattern_needle_movement.py",
    ]
    
    print("=" * 70)
    print("Generating all Chapter 11 animations...")
    print("=" * 70)
    print()
    
    success_count = 0
    failed_scripts = []
    
    for i, script in enumerate(scripts, 1):
        script_path = Path(__file__).parent / script
        
        # Check if script exists
        if not script_path.exists():
            print(f"[{i}/{len(scripts)}] ⚠️  {script} - NOT FOUND")
            failed_scripts.append((script, "File not found"))
            continue
        
        print(f"[{i}/{len(scripts)}] Generating {script}...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout per script
            )
            
            if result.returncode != 0:
                print(f"  ❌ Error in {script}:")
                print(f"     {result.stderr}")
                failed_scripts.append((script, result.stderr))
            else:
                print(f"  ✓ {script} completed")
                if result.stdout:
                    # Print any output from the script (like file paths)
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            print(f"     {line}")
                success_count += 1
        
        except subprocess.TimeoutExpired:
            print(f"  ❌ {script} timed out after 60 seconds")
            failed_scripts.append((script, "Timeout after 60 seconds"))
        
        except Exception as e:
            print(f"  ❌ Unexpected error in {script}: {e}")
            failed_scripts.append((script, str(e)))
        
        print()
    
    # Final summary
    print("=" * 70)
    print(f"Generation complete: {success_count}/{len(scripts)} succeeded")
    print("=" * 70)
    
    if failed_scripts:
        print()
        print("Failed scripts:")
        for script, error in failed_scripts:
            print(f"  • {script}")
            print(f"    {error[:100]}...")
        print()
        sys.exit(1)
    else:
        print()
        print("✅ All animations generated successfully!")
        print()


if __name__ == "__main__":
    main()
