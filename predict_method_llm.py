#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import sys
import csv
import backoff
import openai
import pandas as pd
import tiktoken
import concurrent.futures
import ast  # 追加: メソッド名抽出用
from threading import Lock
from dotenv import load_dotenv

# sklearnのインポート
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# .env読み込み
load_dotenv()

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
INPUT_FILE = "method.csv"  # 変更: 入力ファイル名
OUTPUT_FILE = "predict.csv"
MODEL = "gpt-5.2"  # 必要に応じて gpt-4o などに変更してください

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Error: OPENAI_API_KEY is not set.")
    sys.exit(1)

client = openai.OpenAI(api_key=API_KEY)

# ==========================================
# 2. ユーティリティ関数
# ==========================================
try:
    ENC = tiktoken.get_encoding("o200k_base")
except:
    ENC = tiktoken.get_encoding("cl100k_base")

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if pd.isna(text): return ""
    text = str(text)
    toks = ENC.encode(text, disallowed_special=())
    if len(toks) <= max_tokens:
        return text
    return ENC.decode(toks[:max_tokens])

def extract_method_name_from_code(code: str) -> str:
    """
    Pythonコード文字列から関数名を抽出する
    """
    if pd.isna(code):
        return "unknown"
    try:
        # AST解析で確実に関数名を取得
        tree = ast.parse(str(code))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
    except SyntaxError:
        # コード断片などでパースできない場合の簡易的なフォールバック
        try:
            # "def func_name" のパターンを探す簡易処理
            lines = str(code).splitlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("async def "):
                    part = stripped.split("def ")[1]
                    return part.split("(")[0].strip()
        except:
            pass
    except Exception:
        pass
    return "unknown_method"

# ==========================================
# 3. API予測ロジック
# ==========================================
JSON_SCHEMA = {
    "name": "status_prediction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": { "type": "string" },
            "output": { "type": "integer", "enum": [0, 1] },
            "confidence": { "type": "integer", "minimum": 1, "maximum": 10 }
        },
        "required": ["reason", "output", "confidence"],
        "additionalProperties": False
    }
}

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError), max_tries=5)
def predict_with_gpt(method_code: str) -> tuple[str, int, int]:
    truncated_code = truncate_to_tokens(method_code, 15000)
    
    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert software engineering researcher.\n"
            "Predict if the Python method will be Deleted (0) or Survived (1) in a Pull Request.\n"
            "0: Deleted (redundant, debug, smells), 1: Survived (essential, high quality).\n"
            "Respond in JSON."
        )
    }
    user_prompt = {
        "role": "user",
        "content": f"Code:\n{truncated_code}"
    }
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[system_prompt, user_prompt],
        temperature=0.0,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
    )
    data = json.loads(resp.choices[0].message.content)
    return data["reason"], int(data["output"]), int(data["confidence"])

# ==========================================
# 4. メイン処理
# ==========================================
def process_data(max_workers: int = 10):
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE} with strict CSV parser...")
    try:
        df_input = pd.read_csv(INPUT_FILE, quoting=csv.QUOTE_ALL)
    except Exception as e:
        print(f"Read Error: {e}")
        return

    # ラベル変換
    status_map = {'deleted': 0, 'survived': 1}
    df_input['status_int'] = df_input['status'].astype(str).str.lower().map(status_map).fillna(-1).astype(int)

    # 有効なデータのみ抽出
    df_input = df_input[df_input['status_int'] != -1].reset_index(drop=True)
    
    # -------------------------------------------------------
    # 追加: methodカラムからmethod_nameを抽出
    # -------------------------------------------------------
    print("Extracting method names...")
    df_input['method_name'] = df_input['method'].apply(extract_method_name_from_code)
    
    print(f"Loaded {len(df_input)} valid rows. (Deleted: {len(df_input[df_input['status_int']==0])})")

    # Resume（再開）ロジック
    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            results = pd.read_csv(OUTPUT_FILE).to_dict('records')
            print(f"Resuming from {len(results)} items already in {OUTPUT_FILE}.")
        except:
            print(f"{OUTPUT_FILE} is empty or corrupted. Starting from scratch.")

    # 処理済みインデックスを取得
    processed_indices = {r.get('index_orig') for r in results if 'index_orig' in r}
    
    tasks = []
    for idx, row in df_input.iterrows():
        if idx not in processed_indices:
            # タスクに method_name を追加
            tasks.append((idx, row['status_int'], row['method'], row['method_name']))

    lock = Lock()

    # ラッパー関数も method_name を受け取るように変更
    def task_wrapper(index, original, code, method_name):
        try:
            reason, pred, conf = predict_with_gpt(code)
            with lock:
                results.append({
                    "index_orig": index,
                    "method_name": method_name,  # 追加: カラム保存
                    "original_status": original,
                    "predict": pred,
                    "reason": reason,
                    "confidence": conf,
                    "is_correct": (original == pred)
                })
        except Exception as e:
            print(f"[{index}] Error: {e}")

    # 並列実行
    if tasks:
        print(f"Starting prediction for {len(tasks)} remaining items...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 引数が増えたのでアンパックを変更
            futures = [executor.submit(task_wrapper, i, t, c, n) for i, t, c, n in tasks]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                
                if i % 10 == 0 or i == len(tasks):
                    with lock:
                        current_snapshot = list(results)
                    
                    pd.DataFrame(current_snapshot).to_csv(OUTPUT_FILE, index=False)
                    if i % 50 == 0:
                        print(f"Progress: {i}/{len(tasks)} completed.")

    # 最終保存
    with lock:
        df_final = pd.DataFrame(list(results))
    
    # カラムの順序を整える（オプション）
    columns_order = ["index_orig", "method_name", "original_status", "predict", "confidence", "is_correct", "reason"]
    # 存在するカラムだけを選択して並び替え
    final_cols = [c for c in columns_order if c in df_final.columns]
    df_final = df_final[final_cols]
    
    df_final.to_csv(OUTPUT_FILE, index=False)

    # ==========================================
    # 5. 評価指標の出力
    # ==========================================
    if HAS_SKLEARN and not df_final.empty:
        y_true = df_final['original_status']
        y_pred = df_final['predict']
        y_conf = df_final['confidence']

        y_prob = [ (0.5 + (c/20)) if p==1 else (0.5 - (c/20)) for p, c in zip(y_pred, y_conf)]

        print("\n" + "="*40)
        print(f" 評価レポート (n={len(df_final)})")
        print("="*40)
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.2%}")
        try:
            print(f"AUC      : {roc_auc_score(y_true, y_prob):.4f}")
        except:
            print("AUC      : Undefined")
            
        print(f"F1-Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print("-" * 20)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("="*40)

if __name__ == "__main__":
    process_data(max_workers=10)