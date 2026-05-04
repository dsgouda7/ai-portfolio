#!/usr/bin/env bash
# teardown.sh — Gracefully stop all background processes started by setup.sh
# and deactivate the .venv.
#
# Usage:
#   bash scripts/teardown.sh
#
# Stops (in order):
#   1. Jupyter Lab       (.jupyter.pid  → port 8888)
#   2. MkDocs site       (.mkdocs.pid   → port 8000)
#   3. Ollama server     (.ollama.pid   → port 11434)
#   4. Deactivates the virtual environment if active

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ─── Helpers ──────────────────────────────────────────────────────────────────

step()  { echo; echo "▶ $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ! $*"; }
skip()  { echo "  – $*"; }

# stop_service LABEL PID_FILE FALLBACK_PROC_NAME
#   Sends SIGTERM first, waits up to 3 s, then SIGKILL if still alive.
stop_service() {
    local label="$1"
    local pid_file="$2"
    local fallback_name="${3:-}"

    step "Stopping $label"

    local stopped=false

    # Try PID file first
    if [ -f "$pid_file" ]; then
        local saved_pid
        saved_pid=$(cat "$pid_file" 2>/dev/null | tr -d '[:space:]')
        if [[ "$saved_pid" =~ ^[0-9]+$ ]] && kill -0 "$saved_pid" 2>/dev/null; then
            kill -TERM "$saved_pid" 2>/dev/null || true
            # Wait up to 3 s for graceful exit
            local waited=0
            while kill -0 "$saved_pid" 2>/dev/null && [ "$waited" -lt 30 ]; do
                sleep 0.1
                waited=$((waited + 1))
            done
            if kill -0 "$saved_pid" 2>/dev/null; then
                kill -KILL "$saved_pid" 2>/dev/null || true
                ok "$label (PID $saved_pid) force-killed after timeout"
            else
                ok "$label (PID $saved_pid) stopped"
            fi
            stopped=true
        else
            warn "PID $saved_pid from $(basename "$pid_file") is no longer running"
        fi
        rm -f "$pid_file"
    fi

    # Fallback: find by process name
    if [ "$stopped" = false ] && [ -n "$fallback_name" ]; then
        local pids
        pids=$(pgrep -x "$fallback_name" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs -r kill -TERM 2>/dev/null || true
            sleep 1
            # Force-kill any that survived
            local survivors
            survivors=$(echo "$pids" | xargs -r -I{} sh -c 'kill -0 {} 2>/dev/null && echo {}' || true)
            if [ -n "$survivors" ]; then
                echo "$survivors" | xargs -r kill -KILL 2>/dev/null || true
            fi
            local count
            count=$(echo "$pids" | wc -w | tr -d ' ')
            ok "$label stopped via process name '$fallback_name' ($count process(es))"
            stopped=true
        fi
    fi

    if [ "$stopped" = false ]; then
        skip "$label was not running"
    fi
}

# ─── Banner ───────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  AI/ML Dev Environment Teardown"
echo "  Stopping background processes + venv"
echo "══════════════════════════════════════════════"

# ─── 1. Jupyter Lab ───────────────────────────────────────────────────────────

stop_service "Jupyter Lab" "$REPO_ROOT/.jupyter.pid" "jupyter"

# ─── 2. MkDocs ────────────────────────────────────────────────────────────────

stop_service "MkDocs site" "$REPO_ROOT/.mkdocs.pid" "mkdocs"

# ─── 3. Ollama ────────────────────────────────────────────────────────────────

stop_service "Ollama server" "$REPO_ROOT/.ollama.pid" "ollama"

# ─── 4. Deactivate virtual environment ───────────────────────────────────────

step "Virtual environment"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    venv_name="$(basename "$VIRTUAL_ENV")"
    if declare -f deactivate &>/dev/null; then
        deactivate
        ok "Deactivated venv: $venv_name"
    else
        # Manual cleanup: strip venv bin dir from PATH
        VENV_BIN="$VIRTUAL_ENV/bin"
        export PATH="${PATH//$VENV_BIN:/}"
        export PATH="${PATH//:$VENV_BIN/}"
        unset VIRTUAL_ENV
        ok "Removed venv from PATH: $venv_name"
    fi
else
    skip "No active virtual environment detected (VIRTUAL_ENV not set)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════"
echo "  Teardown complete"
echo "══════════════════════════════════════════════"
echo ""
