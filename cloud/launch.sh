#!/usr/bin/env bash
# Launch an isolated EC2 GPU box for one C2-iteration job. Tags everything for easy teardown.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cloud"

if [[ ! -f config.env ]]; then
  echo "Copy config.env.example → config.env and fill in AWS_* values." >&2
  exit 1
fi
# shellcheck source=/dev/null
source config.env

: "${AWS_REGION:?}"
: "${AWS_KEY_NAME:?}"
: "${AWS_SECURITY_GROUP_ID:?}"
: "${AWS_AMI_ID:?}"
: "${AWS_INSTANCE_TYPE:=g5.xlarge}"

JOB_ID="si-$(date -u +%Y%m%d-%H%M%S)"
ACTIVE_DIR="$ROOT/cloud/.active"
mkdir -p "$ACTIVE_DIR"
JOB_ENV="$ACTIVE_DIR/${JOB_ID}.env"

MARKET=""
if [[ "${AWS_USE_SPOT:-1}" == "1" ]]; then
  MARKET='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
fi

SUBNET_ARG=()
[[ -n "${AWS_SUBNET_ID:-}" ]] && SUBNET_ARG=(--subnet-id "$AWS_SUBNET_ID")

USER_DATA=$(cat <<EOF
#!/bin/bash
set -e
exec > /var/log/safety-interventions-setup.log 2>&1
echo "Job ${JOB_ID} user-data start"
apt-get update -qq && apt-get install -y -qq git
EOF
)

INSTANCE_ID=$(aws ec2 run-instances \
  --region "$AWS_REGION" \
  --image-id "$AWS_AMI_ID" \
  --instance-type "$AWS_INSTANCE_TYPE" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --key-name "$AWS_KEY_NAME" \
  --security-group-ids "$AWS_SECURITY_GROUP_ID" \
  "${SUBNET_ARG[@]}" \
  ${MARKET:+--instance-market-options "$MARKET"} \
  --user-data "$USER_DATA" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Project,Value=safety-interventions},{Key=JobId,Value=${JOB_ID}}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Launched instance $INSTANCE_ID (job $JOB_ID). Waiting for running + public IP..."
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat > "$JOB_ENV" <<EOF
JOB_ID=$JOB_ID
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
AWS_REGION=$AWS_REGION
SSH_USER=ubuntu
REPO_DIR=~/ai_lab/code/safety_interventions
EOF
ln -sf "$JOB_ENV" "$ACTIVE_DIR/latest.env"

echo ""
echo "=== Job $JOB_ID ==="
echo "Instance: $INSTANCE_ID"
echo "IP:       $PUBLIC_IP"
echo "State:    $ACTIVE_DIR/latest.env"
echo ""
echo "Wait ~2 min for SSH, then:"
echo "  cloud/sync_code.sh       # rsync local code (required before first job)"
echo "  cloud/remote_setup.sh    # pip + torch on instance"
echo "  cloud/start_job.sh       # starts pipeline in background"
echo "  cloud/status.sh          # check progress anytime"
echo "  cloud/pull_results.sh    # download artifacts"
echo "  cloud/teardown.sh        # terminate instance"
