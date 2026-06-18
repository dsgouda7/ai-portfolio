import subprocess
import sys

print("Starting compression benchmark...")
print("="*70)

result = subprocess.run(
    [sys.executable, "experiments/run_compression_benchmark.py", "--corpus-type", "both", "--sample-size", "5000"],
    cwd="C:/repos/ai-portfolio/projects/context-optimizer",
    capture_output=True,
    text=True,
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")
