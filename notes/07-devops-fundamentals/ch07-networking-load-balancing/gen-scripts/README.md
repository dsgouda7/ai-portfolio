# Diagram Generation Scripts — Ch.7 Networking & Load Balancing

This directory contains Python scripts to generate diagrams for Chapter 7.

## Scripts

### `gen_ch07_needle_animation.py`
Generates the chapter animation showing load distribution across backends with failure and recovery:
- **Output:** `../img/ch07-networking-load-balancing-needle.gif` (animated), `.png` (static)
- **Frames:** 16 stages showing normal operation, backend2 failure, and recovery
- **Duration:** ~3 seconds (200ms per frame)

### `gen_ch07_reverse_proxy.py`
Generates diagram showing reverse proxy architecture:
- **Output:** `../img/gen_ch07_reverse_proxy.png`
- **Shows:** Client → Firewall → Nginx → Backend pool + Database
- **Highlights:** SSL termination, load balancing, health checks, single entry point

### `gen_ch07_load_balancing_algorithms.py`
Generates diagram comparing three load balancing algorithms:
- **Output:** `../img/gen_ch07_load_balancing_algorithms.png`
- **Algorithms:** 
  - Round-robin (even distribution)
  - Least connections (route to least busy backend)
  - IP hash (sticky sessions)
- **Includes:** Use cases, pros/cons for each algorithm

### `gen_ch07_health_checks.py`
Generates diagram comparing passive vs active health checks:
- **Output:** `../img/gen_ch07_health_checks.png`
- **Shows:** 
  - Passive: Nginx detects failures from real request timeouts
  - Active: Nginx probes `/health` endpoint every 5 seconds
- **Timeline:** Failure detection from t=0 to t=20s
- **Includes:** Configuration examples, pros/cons, best practices

## Running the Scripts

### Prerequisites
```bash
pip install matplotlib pillow numpy
```

### Generate All Diagrams
```bash
python gen_ch07_needle_animation.py
python gen_ch07_reverse_proxy.py
python gen_ch07_load_balancing_algorithms.py
python gen_ch07_health_checks.py
```

### Generate Individual Diagram
```bash
python gen_ch07_reverse_proxy.py
```

## Output Directory

All diagrams are saved to `../img/`:
```
../img/
├── ch07-networking-load-balancing-needle.gif  # Chapter animation
├── ch07-networking-load-balancing-needle.png  # Static version
├── gen_ch07_reverse_proxy.png                  # Reverse proxy architecture
├── gen_ch07_load_balancing_algorithms.png      # Algorithm comparison
└── gen_ch07_health_checks.png                  # Health check strategies
```

## Animation Conventions

Following the established pattern from Math Under the Hood and ML chapters:
- Use `matplotlib.use('Agg')` for headless rendering
- Dark backgrounds (`#1a1a2e`) for animations
- White backgrounds (`#ffffff`) for diagrams
- Color scheme: Blue (clients), Green (Nginx/healthy), Red (failures), Yellow (warnings)
- Save both `.gif` (animated) and `.png` (static) versions of animations

## Diagram Style

- **Clear hierarchy:** Title → Main content → Details
- **Consistent colors:** 
  - Blue (`#3b82f6`): Client/Internet
  - Green (`#059669`): Nginx/Proxy
  - Purple (`#4f46e5`): Backends
  - Red (`#ef4444`): Failures
  - Yellow (`#f59e0b`): Warnings/Configuration
- **Annotations:** Use arrows, labels, and boxes to guide the eye
- **Readability:** 150 DPI, large fonts (10-14pt), high contrast
