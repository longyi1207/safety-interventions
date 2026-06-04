#!/usr/bin/env bash
# Terminate ALL running EC2 instances tagged Project=safety-interventions (or one JOB_ID).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cloud"

if [[ -f config.env ]]; then
  # shellcheck source=/dev/null
  source config.env
fi
AWS_REGION="${AWS_REGION:-us-east-1}"

JOB_FILTER=()
if [[ -n "${1:-}" ]]; then
  JOB_FILTER=(Name=tag:JobId,Values="$1")
  echo "Teardown job: $1"
else
  echo "Teardown ALL Project=safety-interventions instances in $AWS_REGION"
fi

IDS=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:Project,Values=safety-interventions" \
            "Name=instance-state-name,Values=running,pending,stopping,stopped" \
            "${JOB_FILTER[@]}" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

if [[ -z "$IDS" || "$IDS" == "None" ]]; then
  echo "No matching instances."
  exit 0
fi

echo "Terminating: $IDS"
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids $IDS
echo "Done. Update docs/cloud_spend_log.md with actual hours."
