#!/usr/bin/env bash
# Run a command with ai_notes/.env API keys loaded (never prints secrets).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/cloud/_env_from_dotenv.sh"
exec "$@"
