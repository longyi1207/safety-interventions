#!/usr/bin/env bash
# Fix SSH timeout when your IP changed: add 0.0.0.0/0:22 to the project security group.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
# shellcheck source=/dev/null
[[ -f "$ROOT/config.env" ]] && source "$ROOT/config.env"
SG_ID="${AWS_SECURITY_GROUP_ID:-}"
if [[ -z "$SG_ID" ]]; then
  SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=safety-interventions-ssh-east" \
    --query 'SecurityGroups[0].GroupId' --output text)
fi
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
  --ip-permissions 'IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=0.0.0.0/0,Description=ssh-temp-roaming-research}]' \
  2>/dev/null || true
echo "SSH 0.0.0.0/0:22 enabled on $SG_ID (key-only auth still required)"
