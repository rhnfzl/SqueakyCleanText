#!/usr/bin/env bash
# Render resources/<name>.svg to resources/<name>.png at 2x with headless Chrome (white background).
# Usage: scripts/diagrams/render.sh [name ...]   (default: every diagram draw_diagrams.py writes)
set -euo pipefail
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
RES="$(cd "$(dirname "$0")/../../resources" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(sct_overview sct_flow sct_pii_modes)
for name in "${names[@]}"; do
  svg="$RES/$name.svg"
  w=$(sed -n 's/.*<svg[^>]* width="\([0-9]*\)".*/\1/p' "$svg" | head -1)
  h=$(sed -n 's/.*<svg[^>]* height="\([0-9]*\)".*/\1/p' "$svg" | head -1)
  printf '<!doctype html><html><body style="margin:0;background:#fff">%s</body></html>' "$(cat "$svg")" > "$TMP/$name.html"
  "$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --default-background-color=ffffffff --window-size="$w,$h" \
    --screenshot="$RES/$name.png" "file://$TMP/$name.html" 2>/dev/null
  echo "$RES/$name.png (${w}x${h} at 2x)"
done
