#!/bin/bash
# Pre-deploy gate. All must pass.
set -u -o pipefail
cd "$(dirname "$0")/.."
FAIL=0

echo "gate 1: numerical parity (node train/parity.js)"
test -f train/parity_fixture.json || { echo "missing train/parity_fixture.json — run train/export.py"; FAIL=1; }
node train/parity.js | tail -3 || FAIL=1

echo "gate 2: js parse"
node --check site/public/app.js || FAIL=1
node --check site/public/model.js || FAIL=1
node --check site/public/summary.js || FAIL=1
node --check site/public/gan.js || FAIL=1

echo "gate 3: build integrity"
./diabetes-venv/bin/python train/build.py > /tmp/build.log 2>&1 || { echo "build failed"; tail -5 /tmp/build.log; FAIL=1; }

echo "gate 4: vercel config valid JSON"
./diabetes-venv/bin/python -c "import json; json.load(open('deploy/vercel.json'))" || FAIL=1

echo "gate 5: CSP baked into every page"
grep -L 'Content-Security-Policy' deploy/*.html | grep -q . && { echo "missing CSP:"; grep -L 'Content-Security-Policy' deploy/*.html; FAIL=1; }

echo "gate 6: noscript notices on dynamic pages"
grep -q noscript deploy/index.html || { echo "index.html missing noscript"; FAIL=1; }
grep -q noscript deploy/summary.html || { echo "summary.html missing noscript"; FAIL=1; }
grep -q noscript deploy/gan.html || { echo "gan.html missing noscript"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then echo "ALL GATES PASSED"; else echo "FAIL=$FAIL — fix before deploy"; exit 1; fi
