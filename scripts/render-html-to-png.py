"""Render HTML/SVG diagrams to PNG via headless Edge/Chrome.

Usage:
    py scripts/render_html_to_png.py <input.html> <output.png> [width] [height]

Falls back from Edge → Chrome. Requires one of them installed (standard on Windows).
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def pick_browser() -> str:
    for p in (EDGE, CHROME):
        if Path(p).exists():
            return p
    raise SystemExit("No Edge or Chrome found. Install one or update paths in this script.")


def render(html_path: Path, png_path: Path, width: int = 1400, height: int = 920) -> None:
    html_path = html_path.resolve()
    png_path = png_path.resolve()
    browser = pick_browser()
    url = html_path.as_uri()
    args = [
        browser,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={width},{height}",
        "--hide-scrollbars",
        "--default-background-color=ffffff",
        f"--screenshot={png_path}",
        url,
    ]
    print("Running:", " ".join(f'"{a}"' if " " in a else a for a in args))
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise SystemExit(f"browser exited with code {result.returncode}")
    if not png_path.exists():
        raise SystemExit(f"expected output not produced: {png_path}")
    print(f"wrote {png_path}  ({png_path.stat().st_size // 1024} KB)")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    html = Path(sys.argv[1])
    png = Path(sys.argv[2])
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
    h = int(sys.argv[4]) if len(sys.argv) > 4 else 920
    render(html, png, w, h)


if __name__ == "__main__":
    main()
