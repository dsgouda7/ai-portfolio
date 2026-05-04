#!/usr/bin/env bash
# install-hooks.sh — copy hooks from scripts/hooks/ into .git/hooks/
#
# Usage (from repo root):
#   bash scripts/install-hooks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_SOURCE="$SCRIPT_DIR/hooks"
GIT_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DEST="$GIT_ROOT/.git/hooks"

if [ ! -d "$HOOKS_SOURCE" ]; then
  echo "Error: hooks directory not found at $HOOKS_SOURCE"
  exit 1
fi

mkdir -p "$HOOKS_DEST"

installed=0

# Install pre-push hook (rename from pre-push-remove-secrets.sh)
if [ -f "$HOOKS_SOURCE/pre-push-remove-secrets.sh" ]; then
  cp "$HOOKS_SOURCE/pre-push-remove-secrets.sh" "$HOOKS_DEST/pre-push"
  chmod +x "$HOOKS_DEST/pre-push"
  echo "  Installed: pre-push (secret removal)"
  installed=$((installed + 1))
fi

# Install remaining hooks
for hook in "$HOOKS_SOURCE"/*; do
  [ -f "$hook" ] || continue
  name="$(basename "$hook")"
  
  # Skip pre-push-remove-secrets.sh since we already installed it as pre-push
  [ "$name" = "pre-push-remove-secrets.sh" ] && continue
  
  cp "$hook" "$HOOKS_DEST/$name"
  chmod +x "$HOOKS_DEST/$name"
  echo "  Installed: $name"
  installed=$((installed + 1))
done

echo ""
echo "$installed hook(s) installed into .git/hooks/"
echo "Test with: git add . && git commit -m 'test' (on a branch)"
