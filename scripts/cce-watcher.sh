#!/usr/bin/env bash
# cce-watcher.sh — CCE file-system watcher (macOS / Linux)
#
# Keeps the Code Context Engine index up to date between git commits:
#   • Any new file created in the repo  →  cce index --changed-only  (immediate)
#   • Any file deleted from the repo    →  cce index --changed-only  (immediate)
#   • A .ipynb file accumulates ≥ 3 cell additions/changes  →  cce index --changed-only
#
# Usage:  bash scripts/cce-watcher.sh
# The VS Code workspace task "cce-watcher" starts this automatically on folder open.
#
# Dependencies:
#   macOS  — fswatch  (brew install fswatch)
#   Linux  — inotifywait  (apt install inotify-tools  /  yum install inotify-tools)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CCE_BIN="$REPO_ROOT/.venv/bin/cce"
DEBOUNCE_SECS=2

if [ ! -x "$CCE_BIN" ]; then
    echo "cce binary not found at $CCE_BIN — run bash scripts/setup.sh first." >&2
    exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
LAST_TRIGGER=0

cce_index() {
    local now
    now=$(date +%s)
    if (( now - LAST_TRIGGER < DEBOUNCE_SECS )); then return; fi
    LAST_TRIGGER=$now
    "$CCE_BIN" index --changed-only >/dev/null 2>&1 &
}

# Count cells in a .ipynb file using Python (already in the venv)
get_cell_count() {
    local path="$1"
    "$REPO_ROOT/.venv/bin/python3" - "$path" 2>/dev/null <<'EOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        nb = json.load(f)
    print(len(nb.get("cells", [])))
except Exception:
    print(0)
EOF
}

# Paths to skip (regex fragments tested with grep -E)
SKIP_RE='(^|/)(\.(venv|git|cce)|__pycache__|node_modules)(/|$)'

should_skip() { echo "$1" | grep -qE "$SKIP_RE"; }

# Per-notebook state (cell counts and pending deltas) stored in a temp file
WATCH_STATE="$REPO_ROOT/.cce-watch-state.json"

get_state_field() {
    # get_state_field <path> <field>  →  integer
    local key="$1" field="$2"
    if [ ! -f "$WATCH_STATE" ]; then echo 0; return; fi
    "$REPO_ROOT/.venv/bin/python3" - "$WATCH_STATE" "$key" "$field" 2>/dev/null <<'EOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    print(data.get(sys.argv[2], {}).get(sys.argv[3], 0))
except Exception:
    print(0)
EOF
}

set_state_field() {
    # set_state_field <path> <field> <value>
    local key="$1" field="$2" value="$3"
    "$REPO_ROOT/.venv/bin/python3" - "$WATCH_STATE" "$key" "$field" "$value" 2>/dev/null <<'EOF'
import json, sys
path, key, field, value = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    with open(path) as f:
        data = json.load(f)
except Exception:
    data = {}
data.setdefault(key, {})[field] = value
with open(path, "w") as f:
    json.dump(data, f)
EOF
}

handle_create() {
    local path="$1"
    should_skip "$path" && return
    cce_index
}

handle_delete() {
    local path="$1"
    should_skip "$path" && return
    cce_index
}

handle_notebook_change() {
    local path="$1"
    should_skip "$path" && return
    [[ "$path" == *.ipynb ]] || return

    local current_count prev_count pending delta
    current_count=$(get_cell_count "$path")
    prev_count=$(get_state_field "$path" "cell_count")
    pending=$(get_state_field "$path" "pending")

    delta=$(( current_count > prev_count ? current_count - prev_count : prev_count - current_count ))
    set_state_field "$path" "cell_count" "$current_count"
    pending=$(( pending + delta ))

    if (( pending >= 3 )); then
        set_state_field "$path" "pending" 0
        cce_index
    else
        set_state_field "$path" "pending" "$pending"
    fi
}

echo "CCE watcher active — $REPO_ROOT"
echo "  • New file created          → cce index --changed-only"
echo "  • File deleted              → cce index --changed-only"
echo "  • .ipynb ≥ 3 cells changed  → cce index --changed-only"
echo "Press Ctrl+C to stop."

OS="$(uname -s)"

if [[ "$OS" == "Darwin" ]]; then
    # ── macOS: fswatch ────────────────────────────────────────────────────────
    if ! command -v fswatch &>/dev/null; then
        echo "fswatch not found — install with: brew install fswatch" >&2
        exit 1
    fi
    fswatch -r -x \
        --exclude '/\.(venv|git|cce)/' \
        --exclude '/__pycache__/' \
        --exclude '/node_modules/' \
        "$REPO_ROOT" | \
    while read -r path flags; do
        if echo "$flags" | grep -q "Created"; then
            handle_create "$path"
        elif echo "$flags" | grep -q "Removed"; then
            handle_delete "$path"
        elif echo "$flags" | grep -q "Updated"; then
            handle_notebook_change "$path"
        fi
    done

else
    # ── Linux: inotifywait ────────────────────────────────────────────────────
    if ! command -v inotifywait &>/dev/null; then
        echo "inotifywait not found — install with: apt install inotify-tools" >&2
        exit 1
    fi
    inotifywait -m -r \
        -e create \
        -e close_write \
        -e delete \
        --excludei '(\.venv|\.git|__pycache__|\.cce|node_modules)' \
        --format '%w%f %e' \
        "$REPO_ROOT" | \
    while read -r path event; do
        if [[ "$event" == *CREATE* ]]; then
            handle_create "$path"
        elif [[ "$event" == *DELETE* ]]; then
            handle_delete "$path"
        elif [[ "$event" == *CLOSE_WRITE* ]]; then
            handle_notebook_change "$path"
        fi
    done
fi
