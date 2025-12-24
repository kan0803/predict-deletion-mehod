# ベースイメージ
FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール (LightGBM/XGBoost等で必要になる場合があるため)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依存関係ファイルのコピー
COPY requirements.txt .

# Pythonライブラリのインストール
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー (実行時にマウントする場合はコメントアウトでも可)
COPY . .

# デフォルトのコマンド (コンテナ起動時にbashを開く)
CMD ["bash"]