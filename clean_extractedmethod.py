import pandas as pd
import csv
import os
import sys

# === 設定 ===
INPUT_FILE = 'extracted_methods.csv'  # 直前に作成したファイル名
OUTPUT_FILE = 'cleaned_extracted_methods.csv'      # クリーニング後の保存名

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"エラー: 入力ファイル '{INPUT_FILE}' が見つかりません。")
        sys.exit(1)

    print(f"読み込み中: {INPUT_FILE} ...")
    
    # CSV読み込み
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        sys.exit(1)

    total_count = len(df)
    
    # === フィルタリング処理 ===
    # メソッド列が文字列型であることを保証し、"Error:" で始まるものを特定
    # na=False は欠損値(NaN)をFalseとして扱う（念のため）
    error_mask = df['method'].astype(str).str.startswith('Error:')
    
    # エラーの行を抽出（確認用）
    error_df = df[error_mask]
    
    # エラーではない行（正常な行）を抽出
    clean_df = df[~error_mask]

    cleaned_count = len(clean_df)
    removed_count = len(error_df)

    # === 結果出力 ===
    print(f"\n--- 処理結果 ---")
    print(f"元データ件数: {total_count}")
    print(f"除去した件数: {removed_count} (Error行)")
    print(f"残った件数  : {cleaned_count} (正常データ)")

    # エラーの内訳を表示（参考）
    if removed_count > 0:
        print("\n[除去されたエラーの内訳]")
        print(error_df['method'].value_counts().head().to_string())

    # CSV保存
    # quoting=csv.QUOTE_ALL を指定して、改行を含むコードを安全に保存
    clean_df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\n保存完了: {OUTPUT_FILE}")
    print("このファイルを次の分析に使用してください。")

if __name__ == "__main__":
    main()