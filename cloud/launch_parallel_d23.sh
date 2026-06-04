#!/usr/bin/env bash
# Launch 2× g5.xlarge: A=D2-ER, B=D3a+D3c (sequential on B). ~max(5h, 10h) wall clock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cloud"
# shellcheck source=/dev/null
source config.env

launch_one() {
  local TAG="$1"
  local JOB_ID="si-$(date -u +%Y%m%d-%H%M%S)-${TAG}"
  local ACTIVE_DIR="$ROOT/cloud/.active"
  mkdir -p "$ACTIVE_DIR"
  local JOB_ENV="$ACTIVE_DIR/${JOB_ID}.env"

  SUBNET_ARG=()
  [[ -n "${AWS_SUBNET_ID:-}" ]] && SUBNET_ARG=(--subnet-id "$AWS_SUBNET_ID")
  MARKET=""
  if [[ "${AWS_USE_SPOT:-1}" == "1" ]]; then
    MARKET='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
  fi

  local INSTANCE_ID
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
      "ResourceType=instance,Tags=[{Key=Project,Value=safety-interventions},{Key=JobId,Value=${JOB_ID}},{Key=Track,Value=${TAG}}]" \
    --query 'Instances[0].InstanceId' --output text)

  aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
  local PUBLIC_IP
  PUBLIC_IP=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

  cat > "$JOB_ENV" <<EOF
JOB_ID=$JOB_ID
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
AWS_REGION=$AWS_REGION
SSH_USER=ubuntu
REPO_DIR=~/ai_lab/code/safety_interventions
TRACK_TAG=$TAG
EOF
  echo "$JOB_ENV"
  echo "Launched $TAG: $JOB_ID @ $PUBLIC_IP ($INSTANCE_ID)"
}

echo "Launching 2 instances (D2 | D3a+D3c)..."
ENV_D2=$(launch_one "d2")
ENV_D3=$(launch_one "d3")
# shellcheck source=/dev/null
source "$ENV_D2"
echo "D2_JOB=$JOB_ID D2_IP=$PUBLIC_IP" > "$ROOT/cloud/.active/parallel_d23.env"
# shellcheck source=/dev/null
source "$ENV_D3"
echo "D3_JOB=$JOB_ID D3_IP=$PUBLIC_IP" >> "$ROOT/cloud/.active/parallel_d23.env"

echo ""
echo "Wait ~3 min, then:"
echo "  bash cloud/bootstrap_parallel_d23.sh"
