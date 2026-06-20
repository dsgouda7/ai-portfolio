"""
Start Context Optimizer API Gateway (Development Mode)

Run this instead of Docker for faster development iteration.
"""

import sys
from pathlib import Path

# Add src to path
root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))

# Check dependencies
try:
    import fastapi
    import uvicorn
except ImportError:
    print("❌ Missing dependencies. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
    import fastapi
    import uvicorn

if __name__ == "__main__":
    print("="*80)
    print("CONTEXT OPTIMIZER API GATEWAY")
    print("="*80)
    print("\n📡 Starting FastAPI server...")
    print("   http://localhost:8000")
    print("   http://localhost:8000/docs (API documentation)")
    print("   http://localhost:8000/health (health check)")
    print("\n🔄 Auto-reload enabled (development mode)")
    print("   Press Ctrl+C to stop\n")
    print("="*80 + "\n")

    uvicorn.run(
        "docker.gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(root / "src"), str(root / "docker")],
        log_level="info"
    )
