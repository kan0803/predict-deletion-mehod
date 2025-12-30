import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

# === 設定 ===
INPUT_FILE = "predict.csv"
N_ITERATIONS = 100  # 試行回数

def main():
    print(f"Loading {INPUT_FILE}...")
    try:
        # すべてのカラムを読み込む
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit(1)

    # --- データ検証とクリーニング ---
    required_cols = ['original_status', 'predict', 'confidence']
    
    # 必須カラムが存在するかチェック
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The file must contain columns: {required_cols}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    # 数値変換（エラー回避のため）
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # NaNを含む行（変換失敗や欠損）を削除して、必要なカラムだけを残す
    df_clean = df.dropna(subset=required_cols)[required_cols].copy()

    # 整数型にキャスト
    df_clean['original_status'] = df_clean['original_status'].astype(int)
    df_clean['predict'] = df_clean['predict'].astype(int)

    # --- データの分離 ---
    # Deleted (0) と Survived (1) を分離
    df_deleted = df_clean[df_clean['original_status'] == 0]
    df_survived = df_clean[df_clean['original_status'] == 1]
    
    n_deleted = len(df_deleted)
    n_survived = len(df_survived)

    print(f"Original distribution: Deleted(0)={n_deleted}, Survived(1)={n_survived}")

    # 少ない方のクラスに合わせてサンプリング数を決定（通常Deletedが少ないが、逆の場合も考慮）
    n_samples = min(n_deleted, n_survived)
    
    if n_samples == 0:
        print("Error: One of the classes has 0 samples. Cannot proceed with balanced evaluation.")
        sys.exit(1)

    print(f"Sampling size per iteration: {n_samples} each (Total balanced: {n_samples * 2})")

    # --- バランシング評価ループ ---
    scores = []

    for i in range(N_ITERATIONS):
        # 多い方のクラスから、少ない方のクラスと同じ数だけランダム抽出
        if n_deleted < n_survived:
            # Survivedが多い場合
            df_maj_sample = df_survived.sample(n=n_samples, random_state=i)
            df_test = pd.concat([df_deleted, df_maj_sample])
        else:
            # Deletedが多い場合
            df_maj_sample = df_deleted.sample(n=n_samples, random_state=i)
            df_test = pd.concat([df_maj_sample, df_survived])
        
        y_true = df_test['original_status']
        y_pred = df_test['predict']
        
        # 信頼度を用いた確率スコア (Confidence 1-10 -> 確率近似)
        y_prob = [ (0.5 + (c/20)) if p==1 else (0.5 - (c/20)) for p, c in zip(y_pred, df_test['confidence'])]

        scores.append({
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
        })

    # --- 集計結果 ---
    results_df = pd.DataFrame(scores)
    
    print("\n" + "="*45)
    print(f" Balanced Evaluation Summary ({N_ITERATIONS} iterations)")
    print("="*45)
    for col in results_df.columns:
        print(f"{col.capitalize():<10}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")
    print("="*45)

if __name__ == "__main__":
    main()