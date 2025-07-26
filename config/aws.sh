# Macの場合
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# コンピューターのパスワードを要求されます。
# 表示はないですが、入力されていますので、enterを押してください。

# インストールが完了したら、以下のコマンドを実行してください。
aws configure

# IAMユーザーを作成し、アクセスキーを取得します。

# 以下のように入力してください。
# AWS Access Key ID [None]: XXXXXXXXXXXXXXXXXXXX
# AWS Secret Access Key [None]: XXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Default region name [None]: ap-northeast-1  # 東京リージョンなど
# Default output format [None]: json

# バケットの中身を確認する
# aws s3 ls s3://$BUCKET_NAME/sir_hawkes/

# 複数ファイルを一括でダウンロードする
# aws s3 cp s3://$BUCKET_NAME/hawkes_N=10000_z=10/ . --recursive
# 実際の頂点数、平均次数は、自分の実験に合わせて変更してください。