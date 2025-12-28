import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import os
import sys
import warnings

# 警告を無視
warnings.filterwarnings("ignore")

# === 設定 ===
INPUT_FILE = 'pre_features.csv'
OUTPUT_IMPORTANCE_FILE = 'bootstrap_feature_importances.csv'
OUTPUT_REPORT_FILE = 'bootstrap_evaluation_report.txt'

# ブートストラップ設定
N_BOOTSTRAP_ITER = 100 
N_CV_SPLITS = 10        

# ターゲットと除外列
TARGET_COL = 'status'
DROP_COLS = [
    'repo_name', 'pr_url', 'filepath', 'method_key', 'class_name', 'method_name',
    'is_truly_added', 'survival_status'
]

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'n_jobs': -1
}

def load_data(file_path):
    """データ読み込みと基本整形（メタデータを保持）"""
    if not os.path.exists(file_path):
        print(f"エラー: '{file_path}' が見つかりません。")
        sys.exit(1)

    df = pd.read_csv(file_path)
    
    if TARGET_COL not in df.columns:
        print(f"エラー: 目的変数 '{TARGET_COL}' が見つかりません。")
        sys.exit(1)

    valid_statuses = ['survived', 'deleted']
    df = df[df[TARGET_COL].isin(valid_statuses)].copy()

    # メタデータの退避
    meta_cols = [c for c in DROP_COLS if c in df.columns]
    df_meta = df[meta_cols].copy()
    
    # 特徴量処理用
    df_features = df.drop(columns=meta_cols)

    bool_cols = df_features.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_features[col] = df_features[col].astype(int)

    cat_cols = df_features.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        cols_to_encode = [c for c in cat_cols if c != TARGET_COL]
        if cols_to_encode:
            df_features = pd.get_dummies(df_features, columns=cols_to_encode, drop_first=True)

    df_features = df_features.fillna(0)
    df_final = pd.concat([df_meta, df_features], axis=1)
    
    return df_final

def main():
    print(f"--- 処理開始: {N_BOOTSTRAP_ITER}回のブートストラップ検証 (RF vs Random) ---")
    
    df_raw = load_data(INPUT_FILE)
    
    le = LabelEncoder()
    df_raw[TARGET_COL] = le.fit_transform(df_raw[TARGET_COL])
    
    class_names = le.classes_
    print(f"クラス定義: {class_names} (0 -> {class_names[0]}, 1 -> {class_names[1]})")
    
    deleted_val = np.where(class_names == 'deleted')[0][0]
    survived_val = np.where(class_names == 'survived')[0][0]

    df_deleted = df_raw[df_raw[TARGET_COL] == deleted_val]
    df_survived = df_raw[df_raw[TARGET_COL] == survived_val]

    n_deleted = len(df_deleted)
    n_survived = len(df_survived)
    
    print(f"元データ件数: Deleted={n_deleted}, Survived={n_survived}")

    feature_cols = [c for c in df_raw.columns if c not in DROP_COLS and c != TARGET_COL]

    all_rf_results = []
    all_random_results = []
    feature_importances_list = []

    # === ブートストラップループ ===
    for i in range(N_BOOTSTRAP_ITER):
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{N_BOOTSTRAP_ITER}...")

        # 1. アンダーサンプリング (均衡データ作成)
        df_survived_sample = df_survived.sample(n=n_deleted, replace=False, random_state=i)
        df_balanced = pd.concat([df_deleted, df_survived_sample])
        
        # 学習データ準備
        X = df_balanced[feature_cols]
        y = df_balanced[TARGET_COL].values
        
        # 2. スケーリング
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 共通設定
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall'
        }
        # 共通のCV分割を使用
        cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=i)

        # === A. Random Forest モデル ===
        rf_model = RandomForestClassifier(**RF_PARAMS, random_state=i)
        rf_cv_results = cross_validate(rf_model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        rf_iter_summary = {metric: np.mean(rf_cv_results[f'test_{metric}']) for metric in scoring.keys()}
        all_rf_results.append(rf_iter_summary)

        # 特徴量重要度
        rf_model.fit(X_scaled, y)
        feature_importances_list.append(rf_model.feature_importances_)

        # === B. ランダム予測 (Dummy Classifier) ===
        dummy_model = DummyClassifier(strategy='stratified', random_state=i)
        dummy_cv_results = cross_validate(dummy_model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)

        rand_iter_summary = {metric: np.mean(dummy_cv_results[f'test_{metric}']) for metric in scoring.keys()}
        all_random_results.append(rand_iter_summary)

    # === 集計と出力 ===
    rf_results_df = pd.DataFrame(all_rf_results)
    rand_results_df = pd.DataFrame(all_random_results)
    
    print("\n" + "="*80)
    print(f"最終評価結果 (Bootstrap {N_BOOTSTRAP_ITER}回)")
    print("="*80)
    
    print(f"{'Metric':<12} | {'RF Mean':<10} ± {'Std':<8} | {'Random Mean':<10} ± {'Std':<8} | {'Diff (RF-Rand)':<15}")
    print("-" * 90)

    metrics = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    
    with open(OUTPUT_REPORT_FILE, 'w') as f:
        f.write(f"=== Bootstrap Cross-Validation Report ===\n")
        f.write(f"Bootstrap Iterations: {N_BOOTSTRAP_ITER}\n\n")
        f.write("Metric, RF_Mean, RF_Std, Random_Mean, Random_Std, Difference\n")

        for metric in metrics:
            rf_mean = rf_results_df[metric].mean()
            rf_std = rf_results_df[metric].std()
            rand_mean = rand_results_df[metric].mean()
            rand_std = rand_results_df[metric].std()
            diff = rf_mean - rand_mean
            
            print(f"{metric:<12} | {rf_mean:.4f}     ± {rf_std:.4f}   | {rand_mean:.4f}     ± {rand_std:.4f}   | {diff:+.4f}")
            f.write(f"{metric}, {rf_mean:.4f}, {rf_std:.4f}, {rand_mean:.4f}, {rand_std:.4f}, {diff:.4f}\n")

    # 特徴量重要度
    avg_importances = np.mean(feature_importances_list, axis=0)
    std_importances = np.std(feature_importances_list, axis=0)
    
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance_Mean': avg_importances,
        'Importance_Std': std_importances
    }).sort_values(by='Importance_Mean', ascending=False)
    
    imp_df.to_csv(OUTPUT_IMPORTANCE_FILE, index=False)
    print(f"\n完了: {OUTPUT_IMPORTANCE_FILE}, {OUTPUT_REPORT_FILE}")

if __name__ == "__main__":
    main()