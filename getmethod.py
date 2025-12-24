import pandas as pd
import requests
import ast
import csv
import re
import os
import time
import sys
from urllib.parse import quote

# === 設定 ===
INPUT_FILE = 'pre_features.csv'
OUTPUT_FILE = 'extracted_methods.csv'

# === GitHub Tokens ===
# ここに3つのTokenを文字列として入力してください
GITHUB_TOKENS = [
    "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",  # Token 1
    "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",  # Token 2
    "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"   # Token 3
]
VALID_TOKENS = [t for t in GITHUB_TOKENS if t.strip() != ""]
TOKEN_INDEX = 0 

def get_rotated_headers():
    global TOKEN_INDEX
    headers = {
        'User-Agent': 'Mozilla/5.0 (Research Script)',
        'Accept': 'application/vnd.github.v3+json'
    }
    if VALID_TOKENS:
        current_token = VALID_TOKENS[TOKEN_INDEX % len(VALID_TOKENS)]
        headers['Authorization'] = f'token {current_token}'
        TOKEN_INDEX += 1
    return headers

def get_repo_info_and_pr_number(pr_url):
    pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, str(pr_url))
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

def get_pr_base_sha(owner, repo, pr_number):
    """PRのBase SHA（マージ元）のみを取得"""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = get_rotated_headers()
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()['base']['sha']
    except:
        pass
    return None

def get_all_pr_commits_reversed(owner, repo, pr_number):
    """
    PR内の全コミットSHAを「新しい順（降順）」で取得する
    """
    # 1ページあたり100件取得（通常はこれで足りるが、多い場合はページングが必要）
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits?per_page=100"
    headers = get_rotated_headers()
    commits = []
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # dataは通常古い順(Ascending)で来るため、逆順にする
            for item in reversed(data):
                commits.append(item['sha'])
    except Exception as e:
        print(f"  [Error] Commits fetch failed: {e}")
        
    return commits

def fetch_file_content_by_sha(owner, repo, sha, filepath):
    encoded_path = quote(filepath)
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{encoded_path}"
    headers = get_rotated_headers()
    
    try:
        response = requests.get(raw_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return None

def extract_method_ast(source_code, target_method, target_class=None):
    try:
        tree = ast.parse(source_code)
        target_node = None
        
        if target_class and pd.notna(target_class) and str(target_class).strip() != "":
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == target_class:
                    for sub_node in node.body:
                        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if sub_node.name == target_method:
                                target_node = sub_node
                                break
                    if target_node: break
        else:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == target_method:
                        target_node = node
                        break
        
        if target_node:
            lines = source_code.splitlines()
            if hasattr(target_node, 'end_lineno') and target_node.end_lineno is not None:
                start = target_node.lineno - 1
                end = target_node.end_lineno
                return "\n".join(lines[start:end])
            else:
                return "Error: Could not determine end line"
        return None
    except Exception as e:
        return f"Parse Error: {e}"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"エラー: {INPUT_FILE} が見つかりません。")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    total_count = len(df)
    
    if not VALID_TOKENS:
        print("警告: Token未設定。レート制限に注意してください。")

    print(f"処理開始: {total_count} 件")
    print("探索ロジック: PRコミット(新->古) -> Base SHA の順でメソッドを探します。")

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(['status', 'method'])

        success_count = 0
        fail_count = 0

        for index, row in df.iterrows():
            status = row['status']
            method_name = row['method_name']
            class_name = row['class_name']
            filepath = row['filepath']
            pr_url = row['pr_url']

            owner, repo, pr_number = get_repo_info_and_pr_number(pr_url)
            
            print(f"[{index+1}/{total_count}] {method_name} ({status})...", end=" ")
            
            if not pr_number:
                print("Skip (Invalid URL)")
                writer.writerow([status, "Error: Invalid URL"])
                continue

            # 1. 探索対象のコミットリストを作成
            # [最新コミット, ..., 最古コミット]
            search_commits = get_all_pr_commits_reversed(owner, repo, pr_number)
            
            # 2. 最後に Base SHA (PR開始前の状態) も追加
            base_sha = get_pr_base_sha(owner, repo, pr_number)
            if base_sha:
                search_commits.append(base_sha)

            if not search_commits:
                print("Skip (No commits found)")
                writer.writerow([status, "Error: No commits found"])
                time.sleep(1)
                continue

            extracted_code = ""
            found = False
            found_sha = ""

            # 3. リスト順に探索 (見つかったら即終了)
            for sha in search_commits:
                source_code = fetch_file_content_by_sha(owner, repo, sha, filepath)
                if source_code:
                    code_segment = extract_method_ast(source_code, method_name, class_name)
                    # 正常に取得できた場合のみ採用 (Parse Error等はスキップして次のコミットへ)
                    if code_segment and not code_segment.startswith("Error") and not code_segment.startswith("Parse Error"):
                        extracted_code = code_segment
                        found = True
                        found_sha = sha
                        break # ループを抜ける
                
                # API負荷軽減の短い待機
                # time.sleep(0.1) 

            if found:
                print(f"Done. (SHA: {found_sha[:7]})")
                success_count += 1
            else:
                print("Failed (Not found in any commit).")
                extracted_code = "Error: Method not found in history"
                fail_count += 1

            writer.writerow([status, extracted_code])
            
            # PRごとの待機 (コミット数が多いとリクエスト数が増えるため)
            time.sleep(0.5)

    print(f"\n処理完了: {OUTPUT_FILE}")
    print(f"成功: {success_count}, 失敗: {fail_count}")

if __name__ == "__main__":
    main()