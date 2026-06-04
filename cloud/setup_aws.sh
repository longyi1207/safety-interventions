#!/usr/bin/env bash
# One-time AWS prep: EC2 key pair + security group with SSH from your current IP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
MY_IP=$(curl -4 -s ifconfig.me)
KEY_NAME=safety-interventions
SG_NAME=safety-interventions-ssh

mkdir -p "$ROOT/.ssh"
KEY_FILE="$ROOT/.ssh/${KEY_NAME}.pem"

if aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" &>/dev/null; then
  echo "Key pair $KEY_NAME already exists in AWS."
else
  echo "Creating key pair $KEY_NAME..."
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
    --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "Saved private key -> $KEY_FILE"
fi

SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  SG_ID=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$SG_NAME" \
    --description "SSH for safety-interventions cloud jobs" \
    --query GroupId --output text)
  echo "Created security group $SG_ID"
fi

# Idempotent: revoke+authorize would fail if rule exists; try authorize only
aws ec2 authorize-security-group-ingress --region "$REGION" \
  --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32" 2>/dev/null \
  || echo "SSH rule for ${MY_IP}/32 may already exist (OK)"

AMI_ID=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'Images | sort_by(@,&CreationDate) | [-1].ImageId' --output text)

SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
  --filters Name=default-for-az,Values=true \
  --query 'Subnets[0].SubnetId' --output text)

cat > "$ROOT/config.env" <<EOF
export AWS_REGION=$REGION
export AWS_KEY_NAME=$KEY_NAME
export AWS_KEY_FILE=$ROOT/.ssh/${KEY_NAME}.pem
export AWS_SECURITY_GROUP_ID=$SG_ID
export AWS_SUBNET_ID=$SUBNET_ID
export AWS_INSTANCE_TYPE=g5.xlarge
export AWS_USE_SPOT=1
export AWS_AMI_ID=$AMI_ID
export GIT_REPO_URL=https://github.com/longyi1207/ai_lab.git
export GIT_BRANCH=main
export REMOTE_REPO_DIR=~/ai_lab/code/safety_interventions
export CONFIG=configs/qwen7b_harmbench.cloud.yaml
export BOOTSTRAP_JUDGE=0
export CONDITIONS=C0,C2
EOF

echo ""
echo "=== config.env written ==="
cat "$ROOT/config.env"
echo ""
echo "SSH allowed from: ${MY_IP}/32"
echo "AMI: $AMI_ID"
echo "Next: cloud/launch.sh"
