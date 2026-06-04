#!/usr/bin/env bash
# Launch 1× g5.xlarge for D3c only (parallel to D3a on existing instance B).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cloud"
# shellcheck source=/dev/null
source config.env

JOB_ID="si-$(date -u +%Y%m%d-%H%M%S)-d3c"
ACTIVE_DIR="$ROOT/cloud/.active"
mkdir -p "$ACTIVE_DIR"
JOB_ENV="$ACTIVE_DIR/${JOB_ID}.env"

SUBNET_ARG=()
[[ -n "${AWS_SUBNET_ID:-}" ]] && SUBNET_ARG=(--subnet-id "$AWS_SUBNET_ID")
MARKET=""
if [[ "${AWS_USE_SPOT:-1}" == "1" ]]; then
  MARKET='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
fi

INSTANCE_ID=$(aws ec2 run-instances \
  --region "$AWS_REGION" \
  --image-id "$AWS_AMI_ID" \
  --instance-type "${AWS_INSTANCE_TYPE:-g5.xlarge}" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --key-name "$AWS_KEY_NAME" \
  --security-group-ids "$AWS_SECURITY_GROUP_ID" \
  "${SUBNET_ARG[@]}" \
  ${MARKET:+--instance-market-options "$MARKET"} \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Project,Value=safety-interventions},{Key=JobId,Value=${JOB_ID}},{Key=Track,Value=d3c}]" \
  --query 'Instances[0].InstanceId' --output text)

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
TRACK_TAG=d3c
EOF

echo "D3C_JOB=$JOB_ID" > "$ROOT/cloud/.active/parallel_d3c.env"
echo "D3C_IP=$PUBLIC_IP" >> "$ROOT/cloud/.active/parallel_d3c.env"
echo "D3C_INSTANCE_ID=$INSTANCE_ID" >> "$ROOT/cloud/.active/parallel_d3c.env"

echo "Launched D3c: $JOB_ID @ $PUBLIC_IP ($INSTANCE_ID)"
echo "Wait ~3 min, then: bash cloud/bootstrap_d3c.sh"
