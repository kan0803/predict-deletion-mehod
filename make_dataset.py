import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import re
import os

def extract_method_name(code):
    """
    メソッドのソースコードから関数名を抽出する関数。
    'def name' または 'async def name' の形式に対応。
    """
    if not isinstance(code, str):
        return None
    # 正規表現で関数名定義を検索: def name(...)
    match = re.search(r'(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
    if match:
        return match.group(1)
    return None

def make_dataset():
    # 入力ファイル名
    pre_features_file = 'pre_features.csv'
    method_file = 'method.csv'
    
    # ファイルの存在確認
    if not os.path.exists(pre_features_file) or not os.path.exists(method_file):
        print(f"Error: {pre_features_file} または {method_file} が現在のディレクトリに見つかりません。")
        return

    print("Loading datasets...")
    df_pre = pd.read_csv(pre_features_file)
    df_method = pd.read_csv(method_file)

    # 結合のための前処理
    # class_nameの欠損値（NaN）を空文字に置換して統一
    df_pre['class_name'] = df_pre['class_name'].fillna('')
    df_method['class_name'] = df_method['class_name'].fillna('')

    # method.csvの'method'カラム（ソースコード）からメソッド名を抽出
    print("Extracting method names from source code...")
    df_method['extracted_name'] = df_method['method'].apply(extract_method_name)

    # 抽出失敗の確認
    failed_extraction = df_method['extracted_name'].isnull().sum()
    if failed_extraction > 0:
        print(f"Warning: {failed_extraction} 件のメソッド名の抽出に失敗しました。")

    # データセットの結合（内部結合）
    print("Merging datasets...")
    merged = pd.merge(
        df_pre, 
        df_method, 
        left_on=['filepath', 'class_name', 'method_name'], 
        right_on=['filepath', 'class_name', 'extracted_name'],
        suffixes=('', '_method_csv'),
        how='inner'
    )
    
    # --- 【追加箇所】重複削除処理 ---
    print("Checking for duplicates...")
    initial_len = len(merged)
    
    # 重複判定のキーとなるカラム
    subset_cols = ['filepath', 'class_name', 'method_name']
    
    # 重複を削除（keep='first' で最初の1件を残す）
    merged = merged.drop_duplicates(subset=subset_cols, keep='first')
    
    # 削除後の件数確認
    final_len = len(merged)
    dropped_count = initial_len - final_len
    
    if dropped_count > 0:
        print(f"Removed {dropped_count} duplicate rows based on {subset_cols}.")
    else:
        print("No duplicates found.")

    # --- 【重要】インデックスのリセット ---
    # 行を削除するとインデックスが不連続になり、後続のKFoldでのiloc参照でエラーになるためリセットする
    merged = merged.reset_index(drop=True)
    # -----------------------------------

    print(f"Final merged dataframe shape: {merged.shape}")
    
    if len(merged) == 0:
        print("Error: 結合・重複削除後のデータが0件です。")
        return

    # Stratified Split (層化分割)
    target = merged['status']
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    output_dir = 'dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Splitting into 10 folds and saving to /{output_dir}...")
    
    for i, (train_idx, test_idx) in enumerate(skf.split(merged, target)):
        # test_idx をその分割のデータとして使用
        fold_data = merged.iloc[test_idx]
        
        # 1. pre_features側のデータを復元
        cols_pre = df_pre.columns
        fold_pre = fold_data[cols_pre]
        
        # 2. method側のデータを復元
        fold_method = pd.DataFrame()
        
        if 'status_method_csv' in fold_data.columns:
            fold_method['status'] = fold_data['status_method_csv']
        else:
            fold_method['status'] = fold_data['status']
            
        fold_method['method'] = fold_data['method']
        fold_method['filepath'] = fold_data['filepath']
        fold_method['class_name'] = fold_data['class_name']
        
        fold_method = fold_method[['status', 'method', 'filepath', 'class_name']]
        
        # 保存
        fold_pre.to_csv(os.path.join(output_dir, f'pre_features_{i}.csv'), index=False)
        fold_method.to_csv(os.path.join(output_dir, f'method_{i}.csv'), index=False)
        
    print("Done. Dataset splitting complete.")

if __name__ == "__main__":
    make_dataset()