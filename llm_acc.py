import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# === 設定 ===
INPUT_FILE = "predict.csv"
N_ITERATIONS = 100  # 試行回数

def main():
    df = pd.read_csv(INPUT_FILE)
    
    # Deleted (0) と Survived (1) を分離
    df_deleted = df[df['original_status'] == 0]
    df_survived = df[df['original_status'] == 1]
    
    n_samples = len(df_deleted)  # 197件
    print(f"Sampling size: {n_samples} each (Total: {n_samples * 2})")

    scores = []

    for i in range(N_ITERATIONS):
        # SurvivedからDeletedと同じ数だけランダム抽出
        df_survived_sample = df_survived.sample(n=n_samples, random_state=i)
        
        # 評価用均衡データ作成
        df_test = pd.concat([df_deleted, df_survived_sample])
        
        y_true = df_test['original_status']
        y_pred = df_test['predict']
        
        # 信頼度を用いた確率スコア
        y_prob = [ (0.5 + (c/20)) if p==1 else (0.5 - (c/20)) for p, c in zip(y_pred, df_test['confidence'])]

        scores.append({
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob)
        })

    # 集計
    results_df = pd.DataFrame(scores)
    
    print("\n" + "="*45)
    print(f" Balanced Evaluation Summary ({N_ITERATIONS} iterations)")
    print("="*45)
    for col in results_df.columns:
        print(f"{col.capitalize():<10}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")
    print("="*45)

if __name__ == "__main__":
    main()