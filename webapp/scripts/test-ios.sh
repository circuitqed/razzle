#!/usr/bin/env bash
# Run the on-device iOS self-test suite on a simulator, headlessly.
#
# Usage:
#   scripts/test-ios.sh [query]        # e.g. scripts/test-ios.sh 'groups=env,rules'
#   npm run test:ios                   # all groups
#
# Environment:
#   IOS_TEST_DEVICE   simulator name (default "iPhone 17 Pro")
#   IOS_TEST_TIMEOUT  seconds to wait for RESULT (default 900)
#
# Exit code 0 iff the suite prints RESULT: ALL TESTS PASSED.
# Artifacts (log + screenshot) land in webapp/test-results/ios/.
set -euo pipefail

QUERY="${1:-}"
DEVICE="${IOS_TEST_DEVICE:-iPhone 17 Pro}"
TIMEOUT="${IOS_TEST_TIMEOUT:-900}"
BUNDLE_ID="com.lazybrains.knightball"

WEBAPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IOS_DIR="$WEBAPP_DIR/ios/App"
OUT_DIR="$WEBAPP_DIR/test-results/ios"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/native-test.log"
SIMLOG="$OUT_DIR/simulator-stream.log"

step() { echo "==> $*"; }

step "Building web assets"
cd "$WEBAPP_DIR"
npm run build >/dev/null

# Bundle the fixtures' reference model (fixtures.model) when the server still
# has it, so the inference group gets Python-reference depth without network.
# The suite degrades gracefully (GPU-vs-CPU agreement only) when absent.
CACHE="$WEBAPP_DIR/node_modules/.cache/knightball"
mkdir -p "$CACHE"
FIXMODEL=$(python3 -c 'import json; print(json.load(open("public/inference-fixtures.json"))["model"])')
if [ ! -f "$CACHE/fixtures-$FIXMODEL.onnx" ]; then
  step "Fetching fixtures model $FIXMODEL (one-time)"
  if INFO=$(curl -sf "https://knightball.org/api/models/onnx/by-name/$FIXMODEL"); then
    URLPATH=$(echo "$INFO" | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])')
    curl -sf "https://knightball.org/api${URLPATH}" -o "$CACHE/fixtures-$FIXMODEL.onnx" || true
  else
    echo "    fixtures model '$FIXMODEL' not on server — reference checks will be skipped"
  fi
fi
if [ -f "$CACHE/fixtures-$FIXMODEL.onnx" ]; then
  cp "$CACHE/fixtures-$FIXMODEL.onnx" dist/test-model-fixtures.onnx
fi

step "Syncing Capacitor"
npx cap sync ios >/dev/null

step "Pointing app entry at the test suite${QUERY:+ (?$QUERY)}"
python3 - "$QUERY" <<'EOF'
import sys
q = sys.argv[1]
p = 'ios/App/App/public/index.html'
html = open(p).read()
url = '/test-native.html' + ('?' + q if q else '')
html = html.replace('</title>', f'</title>\n    <meta http-equiv="refresh" content="0;url={url}" />')
open(p, 'w').write(html)
EOF

step "Booting simulator: $DEVICE"
if [[ "$DEVICE" =~ ^[0-9A-F-]{36}$ ]]; then
  UDID="$DEVICE"
else
  UDID=$(xcrun simctl list devices available | grep "$DEVICE (" | head -1 | grep -oE '[0-9A-F-]{36}')
fi
[ -n "$UDID" ] || { echo "No available simulator named '$DEVICE'"; exit 1; }
xcrun simctl bootstatus "$UDID" -b >/dev/null

step "Building app for simulator"
cd "$IOS_DIR"
xcodebuild -project App.xcodeproj -scheme App \
  -packageAuthorizationProvider netrc \
  -destination "id=$UDID" -derivedDataPath build build 2>&1 |
  grep -E "error:|BUILD (SUCCEEDED|FAILED)" || true

APP=$(find build/Build/Products -name "App.app" | head -1)
[ -n "$APP" ] || { echo "Build failed: no App.app"; exit 1; }

step "Installing + launching"
xcrun simctl install "$UDID" "$APP"
xcrun simctl terminate "$UDID" "$BUNDLE_ID" 2>/dev/null || true

# Capacitor bridges JS console via Swift print() → app stdout, which only
# --console-pty exposes (os_log streaming never sees console messages).
: > "$SIMLOG"
xcrun simctl launch --console-pty "$UDID" "$BUNDLE_ID" > "$SIMLOG" 2>&1 &
LAUNCH_PID=$!
trap 'kill $LAUNCH_PID 2>/dev/null || true' EXIT

step "Waiting for RESULT (timeout ${TIMEOUT}s)"
DEADLINE=$(( $(date +%s) + TIMEOUT ))
while ! grep -q "\[native-test\] RESULT:" "$SIMLOG"; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "TIMED OUT waiting for suite result"
    xcrun simctl io "$UDID" screenshot "$OUT_DIR/timeout.png" >/dev/null 2>&1 || true
    exit 2
  fi
  sleep 5
done

grep -o "\[native-test\].*" "$SIMLOG" | sed 's/\[native-test\] //' > "$LOG"
xcrun simctl io "$UDID" screenshot "$OUT_DIR/final.png" >/dev/null 2>&1 || true

# Restore the synced index.html (drop the test redirect)
cd "$WEBAPP_DIR" && npx cap sync ios >/dev/null

echo
echo "──────────────── suite output ────────────────"
cat "$LOG"
echo "──────────────────────────────────────────────"
echo "Artifacts: $LOG, $OUT_DIR/final.png"

grep -q "RESULT: ALL TESTS PASSED" "$LOG"
