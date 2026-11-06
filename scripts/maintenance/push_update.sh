#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

msg="${1:-"Update $(date +'%Y-%m-%d %H:%M:%S')"}"

branch="$(git symbolic-ref --short HEAD)" || {
  echo "A named branch is required before committing and pushing." >&2
  exit 1
}

git -c core.ignorecase=false add -A

# Block large files (>=90MB) from being committed
if ! python - <<'PY'
import os, subprocess, sys
limit = 90 * 1024 * 1024
res = subprocess.run(['git','diff','--cached','--name-only','-z'], stdout=subprocess.PIPE, check=True)
files = [f.decode() for f in res.stdout.split(b'\0') if f]
big = []
for p in files:
    if os.path.exists(p) and os.path.getsize(p) >= limit:
        big.append(p)
if big:
    print("ERROR: staged files >= 90MB:", file=sys.stderr)
    for p in big[:50]:
        print(p, file=sys.stderr)
    sys.exit(2)
PY
then
  git reset
  exit 1
fi

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$msg"
git push -u origin "${branch}"
