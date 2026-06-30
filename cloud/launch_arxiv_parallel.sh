#!/usr/bin/env bash
# Launch 4× GPU for arXiv MVA tracks A–D in parallel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cloud"
# shellcheck source=/dev/null
source config.env

SUBNET="${AWS_SUBNET_ID:-subnet-0719033ec0ad577b4}"
ITYPE="${AWS_INSTANCE_TYPE:-g4dn.xlarge}"

launch_one() {
  local TAG="$1"
  local JOB_ID="si-$(date -u +%Y%m%d-%H%M%S)-arxiv-${TAG}"
  local ACTIVE_DIR="$ROOT/cloud/.active"
  mkdir -p "$ACTIVE_DIR"
  local JOB_ENV="$ACTIVE_DIR/${JOB_ID}.env"

  MARKET=""
  if [[ "${AWS_USE_SPOT:-0}" == "1" ]]; then
    MARKET='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
  fi

  local INSTANCE_ID
  INSTANCE_ID=$(aws ec2 run-instances \
    --region "$AWS_REGION" \
    --image-id "$AWS_AMI_ID" \
    --instance-type "$ITYPE" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --key-name "$AWS_KEY_NAME" \
    --security-group-ids "$AWS_SECURITY_GROUP_ID" \
    --subnet-id "$SUBNET" \
    ${MARKET:+--instance-market-options "$MARKET"} \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Project,Value=safety-interventions},{Key=JobId,Value=${JOB_ID}},{Key=Track,Value=arxiv-${TAG}}]" \
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
TRACK_TAG=arxiv-${TAG}
SUBNET_ID=$SUBNET
INSTANCE_TYPE=$ITYPE
EOF
  echo "Launched arxiv-${TAG}: $JOB_ID @ $PUBLIC_IP ($INSTANCE_ID) $ITYPE" >&2
  sleep 8
  echo "$JOB_ENV"
}

echo "Launching 4× $ITYPE on-demand in $SUBNET ..."
ENV_A=$(launch_one "a")
ENV_B=$(launch_one "b")
ENV_C=$(launch_one "c")
ENV_D=$(launch_one "d")

PAR="$ROOT/cloud/.active/parallel_arxiv.env"
{
  # shellcheck source=/dev/null
  source "$ENV_A"; echo "ARXIV_A_JOB=$JOB_ID"; echo "ARXIV_A_IP=$PUBLIC_IP"; echo "ARXIV_A_ENV=$ENV_A"
  # shellcheck source=/dev/null
  source "$ENV_B"; echo "ARXIV_B_JOB=$JOB_ID"; echo "ARXIV_B_IP=$PUBLIC_IP"; echo "ARXIV_B_ENV=$ENV_B"
  # shellcheck source=/dev/null
  source "$ENV_C"; echo "ARXIV_C_JOB=$JOB_ID"; echo "ARXIV_C_IP=$PUBLIC_IP"; echo "ARXIV_C_ENV=$ENV_C"
  # shellcheck source=/dev/null
  source "$ENV_D"; echo "ARXIV_D_JOB=$JOB_ID"; echo "ARXIV_D_IP=$PUBLIC_IP"; echo "ARXIV_D_ENV=$ENV_D"
} > "$PAR"

echo ""
echo "Wrote $PAR"
