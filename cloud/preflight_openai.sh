#!/usr/bin/env bash
# Exit 0 if OPENAI_API_KEY works for a minimal API call.
set -euo pipefail
PY="${PY:-python3}"
"$PY" - <<'PY'
import os, sys
key = os.environ.get("OPENAI_API_KEY", "")
if not key or len(key) < 20:
    sys.exit(1)
from openai import OpenAI
try:
    OpenAI(api_key=key).models.list()
except Exception:
    sys.exit(1)
print("openai_ok")
PY
