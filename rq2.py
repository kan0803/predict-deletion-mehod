#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
import warnings
import json
import math
import backoff
import openai
import tiktoken
import concurrent.futures
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 警告抑制
warnings.filterwarnings("ignore")
load_dotenv()

# ==========================================
# 設定
# ==========================================
DATASET_DIR = "dataset"
N_FOLDS = 10
GPT_MODEL = "gpt-4o" 

# RFハイパーパラメータ
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'n_jobs': -1,
    'random_state': 42
}

# 除外カラム
DROP_COLS = [
    'repo_name', 'pr_url', 'filepath', 'class_name', 'method_name',
    'is_truly_added', 'survival_status', 'avg_var_name_length', 'method_key', 'status', 'extracted_name'
]

STATUS_MAP = {'deleted': 0, 'survived': 1}
TARGET_CLASS = 0 

# ==========================================
# 共通関数
# ==========================================
def calculate_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except ValueError:
        auc = 0.5

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': auc,
        'f1': f1_score(y_true, y_pred, pos_label=TARGET_CLASS, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=TARGET_CLASS, zero_division=0),
        'precision': precision_score(y_true, y_pred, pos_label=TARGET_CLASS, zero_division=0)
    }

def print_metrics(name, metrics):
    print(f"[{name}] (Target Class: Deleted/0)")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")

def load_fold_data(fold_idx, file_prefix):
    path = os.path.join(DATASET_DIR, f"{file_prefix}_{fold_idx}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def preprocess_data(df, is_train=False):
    # ターゲット変換
    df['status_int'] = df['status'].map(STATUS_MAP)
    df = df.dropna(subset=['status_int'])
    
    # 学習データのみアンダーサンプリング
    if is_train:
        df_del = df[df['status_int'] == 0]
        df_sur = df[df['status_int'] == 1]
        
        n_del = len(df_del)
        n_sur = len(df_sur)
        
        if n_del < n_sur:
            df_sur = df_sur.sample(n=n_del, replace=False, random_state=42)
        else:
            df_del = df_del.sample(n=n_sur, replace=False, random_state=42)
            
        df = pd.concat([df_del, df_sur])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    y = df['status_int'].values
    
    feature_cols = [c for c in df.columns if c not in DROP_COLS and c != 'status_int']
    X = df[feature_cols]
    
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    
    return X, y

# ==========================================
# 1. Random Forest (Feature Importance追加版)
# ==========================================
def run_rf_experiment():
    print("\n" + "="*60)
    print("1. Random Forest Experiment (Undersampled Train)")
    print("="*60)
    
    all_scores = []
    feature_importances_list = [] # 各Foldの重要度を格納するリスト
    
    for test_idx in range(N_FOLDS):
        df_test_raw = load_fold_data(test_idx, 'pre_features')
        if df_test_raw is None: continue
        
        train_dfs = []
        for train_idx in range(N_FOLDS):
            if train_idx == test_idx: continue
            tmp = load_fold_data(train_idx, 'pre_features')
            if tmp is not None:
                train_dfs.append(tmp)
        df_train_raw = pd.concat(train_dfs, ignore_index=True)
        
        X_train, y_train = preprocess_data(df_train_raw, is_train=True)
        X_test, y_test = preprocess_data(df_test_raw, is_train=False)
        
        # カラムアライメント
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train_scaled, y_train)
        
        # --- 重要度の記録 ---
        # カラム名と重要度をペアにして保存
        imp_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        })
        feature_importances_list.append(imp_df)
        
        y_pred = rf.predict(X_test_scaled)
        y_prob = rf.predict_proba(X_test_scaled)[:, 1]
        
        scores = calculate_metrics(y_test, y_pred, y_prob)
        all_scores.append(scores)
        print(f"Fold {test_idx}: Acc={scores['accuracy']:.4f}, F1(Del)={scores['f1']:.4f}")

    # --- 平均スコアの算出 ---
    avg_scores = pd.DataFrame(all_scores).mean().to_dict()
    print("-" * 30)
    print_metrics("RF Average", avg_scores)

    # --- 平均特徴量重要度の算出と保存 ---
    if feature_importances_list:
        all_imp = pd.concat(feature_importances_list)
        # 特徴量ごとに平均と標準偏差を計算
        avg_imp = all_imp.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
        avg_imp = avg_imp.sort_values(by='mean', ascending=False)
        
        output_file = 'rf_feature_importance.csv'
        avg_imp.to_csv(output_file, index=False)
        print(f"\n[Feature Importance] Saved to {output_file}")
        print("Top 10 Features:")
        print(avg_imp.head(10).to_string(index=False))

    return avg_scores

# ==========================================
# 2. Random Prediction (Uniform)
# ==========================================
def run_random_experiment():
    print("\n" + "="*60)
    print("2. Random Prediction Experiment (Uniform 50/50)")
    print("="*60)
    
    all_scores = []
    
    for idx in range(N_FOLDS):
        df = load_fold_data(idx, 'pre_features')
        if df is None: continue
        
        df['status_int'] = df['status'].map(STATUS_MAP)
        df = df.dropna(subset=['status_int'])
        y_true = df['status_int'].values
        X_dummy = np.zeros((len(y_true), 1))
        
        dummy = DummyClassifier(strategy='uniform', random_state=idx)
        dummy.fit(X_dummy, y_true) 
        
        y_pred = dummy.predict(X_dummy)
        y_prob = dummy.predict_proba(X_dummy)[:, 1]
        
        scores = calculate_metrics(y_true, y_pred, y_prob)
        all_scores.append(scores)
        
    avg_scores = pd.DataFrame(all_scores).mean().to_dict()
    print("-" * 30)
    print_metrics("Random Average", avg_scores)
    return avg_scores

# ==========================================
# 3. GPT Experiment (gpt-4o) with Logprobs & Reason
# ==========================================
try:
    ENC = tiktoken.get_encoding("o200k_base")
except:
    ENC = tiktoken.get_encoding("cl100k_base")

def truncate_to_tokens(text, max_tokens=15000):
    if pd.isna(text): return ""
    text = str(text)
    toks = ENC.encode(text, disallowed_special=())
    if len(toks) <= max_tokens: return text
    return ENC.decode(toks[:max_tokens])

JSON_SCHEMA = {
    "name": "status_prediction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": { "type": "string" },
            "output": { "type": "integer", "enum": [0, 1] }
        },
        "required": ["reason", "output"],
        "additionalProperties": False
    }
}

def calculate_true_probability(logprobs_content, predicted_val):
    """
    Logprobsから '0' (Deleted) である確率を厳密に計算する。
    """
    if not logprobs_content:
        return 0.5

    # 1. 出力トークン(predicted_val)に対応するLogprobエントリを特定
    # JSONの "output": 0 または "output": 1 の部分を探す
    target_token_index = -1
    str_val = str(predicted_val)
    
    # 後ろから探索（JSONの構造上、outputは最後の方に来るため）
    for i in range(len(logprobs_content) - 1, -1, -1):
        token_str = logprobs_content[i].token.strip()
        if token_str == str_val:
            target_token_index = i
            break
    
    if target_token_index == -1:
        return 0.5 

    # 2. その位置での Top Logprobs を取得
    top_logprobs = logprobs_content[target_token_index].top_logprobs
    
    lp_0 = -9999.0
    lp_1 = -9999.0
    
    found = False
    for item in top_logprobs:
        t_str = item.token.strip()
        if t_str == '0':
            lp_0 = item.logprob
            found = True
        elif t_str == '1':
            lp_1 = item.logprob
            found = True
            
    if not found:
        # Top候補に0も1もいない場合（稀）、予測値を信じる
        return 1.0 if predicted_val == 0 else 0.0

    # 3. Softmax で確率計算
    try:
        prob_0 = math.exp(lp_0)
        prob_1 = math.exp(lp_1)
        total = prob_0 + prob_1
        
        if total == 0: return 0.5
        return prob_0 / total
        
    except OverflowError:
        return 1.0 if lp_0 > lp_1 else 0.0

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError), max_tries=5)
def predict_with_gpt(client, code):
    """
    戻り値: (output_label, deleted_probability, reason_text)
    """
    try:
        code_trunc = truncate_to_tokens(code)
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Predict if the method will be Deleted(0) or Survived(1). Briefly explain the reason. JSON response."},
                {"role": "user", "content": f"Code:\n{code_trunc}"}
            ],
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            
            # === 重要: Logprobs取得設定 ===
            logprobs=True,
            top_logprobs=20 
        )
        
        # 1. コンテンツのパース
        content = resp.choices[0].message.content
        d = json.loads(content)
        pred_output = d["output"]
        pred_reason = d.get("reason", "") # 理由を取得
        
        # 2. Logprobsから確率計算
        logprobs_data = resp.choices[0].logprobs.content
        prob_deleted = calculate_true_probability(logprobs_data, pred_output)
        
        return pred_output, prob_deleted, pred_reason

    except Exception as e:
        print(f"GPT API Error: {e}")
        return -1, -1, ""

def run_gpt_experiment():
    print("\n" + "="*60)
    print(f"3. GPT Experiment ({GPT_MODEL}) with Logprobs & Reason")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipped: OPENAI_API_KEY not found.")
        return None

    client = openai.OpenAI(api_key=api_key)
    os.makedirs("gpt_results", exist_ok=True)
    
    all_scores = []
    total_samples = 0
    total_dropped = 0
    
    for idx in range(N_FOLDS):
        print(f"Processing Fold {idx}...")
        df = load_fold_data(idx, 'method')
        if df is None: continue
        
        result_file = f"gpt_results/pred_fold_{idx}_logprobs.csv"
        
        if os.path.exists(result_file):
            print(f"  Loading existing results from {result_file}")
            df_res = pd.read_csv(result_file)
        else:
            predictions = {}
            # 並列処理 (RateLimitに注意して調整してください)
            with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
                futures = {executor.submit(predict_with_gpt, client, row['method']): i for i, row in df.iterrows()}
                
                for f in concurrent.futures.as_completed(futures):
                    i = futures[f]
                    try:
                        out, prob, reason = f.result()
                        predictions[i] = (out, prob, reason)
                    except Exception as e:
                        print(f"Err {i}: {e}")
                        predictions[i] = (-1, -1, "")

            # 結果をDataFrameにマッピング
            df['predict'] = df.index.map(lambda x: predictions.get(x, (-1, -1, ""))[0])
            df['prob_deleted'] = df.index.map(lambda x: predictions.get(x, (-1, -1, ""))[1])
            df['reason'] = df.index.map(lambda x: predictions.get(x, (-1, -1, ""))[2])
            
            # 保存
            df.to_csv(result_file, index=False)
            df_res = df
        
        # --- 評価 ---
        total_samples += len(df_res)
        
        # エラー行 (-1) の除外
        valid_mask = (df_res['predict'] != -1)
        df_valid = df_res[valid_mask].copy()
        
        dropped_count = len(df_res) - len(df_valid)
        total_dropped += dropped_count
        
        if len(df_valid) == 0:
            print(f"  Fold {idx}: No valid predictions.")
            continue

        y_true = df_valid['status'].map(STATUS_MAP).astype(int)
        y_pred = df_valid['predict'].astype(int)
        y_prob = df_valid['prob_deleted'] # AUC用確率
        
        scores = calculate_metrics(y_true, y_pred, y_prob)
        all_scores.append(scores)
        print(f"  Fold {idx}: Acc={scores['accuracy']:.4f}, AUC={scores['auc']:.4f}, F1={scores['f1']:.4f}")

    if not all_scores:
        print("No valid results.")
        return None

    avg_scores = pd.DataFrame(all_scores).mean().to_dict()
    print("-" * 30)
    print_metrics("GPT Average (Logprobs)", avg_scores)
    
    print("-" * 30)
    print(f"Total Samples: {total_samples}, Dropped: {total_dropped}")
    if total_samples > 0:
        print(f"Drop Rate: {total_dropped/total_samples:.2%}")
    print("-" * 30)

    return avg_scores

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    rf_res = run_rf_experiment()
    rand_res = run_random_experiment()
    gpt_res = run_gpt_experiment()
    
    print("\n" + "="*80)
    print(f"Target Class: Deleted (0)")
    print(f"{'Metric':<12} | {'RF':<10} | {'Random':<10} | {'GPT-4o':<10}")
    print("-" * 80)
    for m in ['accuracy', 'auc', 'f1', 'recall', 'precision']:
        rf_val = rf_res[m] if rf_res else 0
        rand_val = rand_res[m] if rand_res else 0
        gpt_val = gpt_res[m] if gpt_res else 0
        print(f"{m:<12} | {rf_val:.4f}      | {rand_val:.4f}      | {gpt_val:.4f}")
    print("="*80)