#!/usr/bin/env bash
# Load API keys from repo-root .env (ai_notes/.env). Never echo values.
set -euo pipefail
_SCRIPT="${BASH_SOURCE[0]:-$0}"
_CLOUD_DIR="$(cd "$(dirname "$_SCRIPT")" && pwd)"
_SI_ROOT="$(cd "$_CLOUD_DIR/.." && pwd)"
_AI_NOTES_ROOT="$(cd "$_SI_ROOT/../.." && pwd)"
_ENV_FILE="${AI_NOTES_ENV:-$_AI_NOTES_ROOT/.env}"

if [[ ! -f "$_ENV_FILE" ]]; then
  return 1 2>/dev/null || exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" =~ ^[[:space:]]*$ ]] && continue
  if [[ "$line" =~ ^(OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY|HUGGING_FACE_TOKEN)= ]]; then
    export "$line"
  fi
done < "$_ENV_FILE"
