#!/bin/bash

# pemファイルは${KEY_PATH}/にあるものとする。

# 1. 環境変数の読み込み
set -a
source .env
set +a

# 2. 秘密鍵のパーミッション設定（※1回だけでOK）
chmod 400 ${KEY_PATH}/${YOUR_KEY}.pem

# 3. コードをEC2へアップロード（リモートのホームディレクトリへ）
rsync -avz -e "ssh -i ${KEY_PATH}/${YOUR_KEY}.pem" \
  simulation s3 scripts requirements.txt .env \
  ${EC2_USER}@${EC2_HOST}:sir-hawkes

# 4. EC2にSSH接続し、Pythonコードを実行し、環境変数を使う
ssh -t -i ${KEY_PATH}/${YOUR_KEY}.pem ${EC2_USER}@${EC2_HOST} << EOF
    sudo yum install -y python3-pip
    cd sir-hawkes
    set -a
    source .env
    set +a
    pip install -r requirements.txt --user
    export BUCKET_NAME=${BUCKET_NAME}
    python3 -u scripts/run.py

    echo "[INFO] Simulation completed. Proceeding to cleanup..."
    cd ..
    rm -rf sir-hawkes
    echo "[INFO] Cleanup completed."
EOF
