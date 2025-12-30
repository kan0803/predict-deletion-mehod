import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
CSV_FILE_PATH = 'pre_features.csv'
TARGET_COL = 'status'

TARGET_FEATURES_LIST = [
    'code_loc', 'char_length', 'tokens', 'docstring_words', 'method_name_words',
    'is_getter', 'is_setter', 'is_IsHas', 'is_test', 'is_in_test_code',
    'is_private', 'is_dunder_method', 'has_return', 'number_of_variable',
    'number_of_print', 'cyclomatic_complexity', 'halstead_volume',
    'max_nesting_depth', 'param_count', 'call_expression_count',
    'comment_ratio', 'uses_try_except', 'uses_constants'
]

# Top 3 選定時に除外する「2値変数」のプレフィックス (これらを含む変数はTop3候補から外れます)
BINARY_PREFIXES = []

# ランキング計算自体から外すもの
IGNORE_FOR_STATS_RANKING = ['is_truly_added']

# 外れ値の上限設定
outlier_caps = {
    'comment_loc_ratio': 0.4,
    'char_length': 6000,
    'loc': 150,
    'docstring_length': 500,
    'cyclomatic_complexity': 20,
    'halstead_volume': 1000,
    'call_expression_count': 20,
}

TOP_N = 3

# 文字サイズ設定
FONT_SIZES = {
    'title': 55,      # タイトル
    'xlabel': 55,     # X軸ラベル
    'ylabel': 55,     # Y軸目盛
    'd_val': 55       # 効果量表示
}

plt.rcdefaults()
plt.rcParams.update({'font.size': 40}) 

# ---------------------------------------------------------
# 関数
# ---------------------------------------------------------
def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    if n1 < 2 or n2 < 2 or (var1 == 0 and var2 == 0):
        return 0
    pool_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pool_std == 0:
        return 0
    d = (np.mean(group1) - np.mean(group2)) / pool_std
    return d

# ---------------------------------------------------------
# データ処理
# ---------------------------------------------------------
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"データを読み込みました: {CSV_FILE_PATH}")
except FileNotFoundError:
    print("【注意】CSVファイルが見つかりません。ダミーデータを作成します...")
    np.random.seed(42)
    data = {'status': np.random.choice(['deleted', 'survived'], 300)}
    for col in TARGET_FEATURES_LIST:
        if col in ['first_word', 'definition_location']:
            data[col] = np.random.choice(['dummy_str', 'dummy_val'], 300)
        elif any(col.startswith(pre) for pre in BINARY_PREFIXES):
            data[col] = np.random.choice([0, 1], 300)
        else:
            data[col] = np.random.exponential(10, 300)
    df = pd.DataFrame(data)

all_feature_cols = [
    col for col in df.columns 
    if col in TARGET_FEATURES_LIST 
    and pd.api.types.is_numeric_dtype(df[col])
]

# ---------------------------------------------------------
# 統計検定 & Top3選定 (出力処理を復元)
# ---------------------------------------------------------
d_values = {}
p_values = {}
groups = df[TARGET_COL].unique()

# ランキング候補（除外指定および2値変数以外）
ranking_candidates = [
    c for c in all_feature_cols 
    if c not in IGNORE_FOR_STATS_RANKING
    and not any(c.startswith(pre) for pre in BINARY_PREFIXES)
]

print("\n" + "="*60)
print("【統計検定結果 (Mann-Whitney U) & 効果量 (Cohen's d)】")
print("="*60)

top_features = []

if len(groups) == 2:
    group_a_label = groups[0]
    group_b_label = groups[1]
    group_a = df[df[TARGET_COL] == group_a_label]
    group_b = df[df[TARGET_COL] == group_b_label]

    print(f"Group A: {group_a_label} (n={len(group_a)})")
    print(f"Group B: {group_b_label} (n={len(group_b)})\n")

    # 全特徴量について計算し、結果を表示
    for col in all_feature_cols:
        d = calculate_cohens_d(group_a[col], group_b[col])
        d_values[col] = abs(d)
        
        stat, p = mannwhitneyu(group_a[col], group_b[col], alternative='two-sided')
        p_values[col] = p
        
        # 有意差のスター
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        
        # Top3候補外かどうかの注釈
        is_excluded = (col not in ranking_candidates)
        note = " (Top3対象外)" if is_excluded else ""

        # ★復元箇所: コンソールへの出力
        print(f"{col:<25} | d={abs(d):.3f} | p={p:.3e} {star}{note}")

    # Top3選定 (有意かつ効果量が大きい順)
    valid_candidates = [c for c in ranking_candidates if p_values[c] < 0.05]
    valid_candidates.sort(key=lambda x: d_values[x], reverse=True)
    top_features = valid_candidates[:TOP_N]
else:
    print("ターゲットクラスが2つではありません。")
    top_features = ranking_candidates[:TOP_N]

# ---------------------------------------------------------
# Top 3 プロット
# ---------------------------------------------------------
print("\n" + "="*60)
print(f"Top {len(top_features)} 特徴量をプロットします: {top_features}")

df_capped = df.copy()
for col, limit in outlier_caps.items():
    if col in df_capped.columns:
        df_capped.loc[df_capped[col] > limit, col] = limit

custom_palette = {'deleted': 'red', 'survived': 'C0'} 
order_list = ['survived', 'deleted']

if len(top_features) > 0:
    fig, axes = plt.subplots(1, len(top_features), figsize=(30, 12))
    if len(top_features) == 1: axes = [axes]

    for i, col in enumerate(top_features):
        sns.violinplot(
            data=df_capped, x=pd.Series(0, index=df_capped.index), y=col,
            hue=TARGET_COL, hue_order=order_list,
            split=True, ax=axes[i], palette=custom_palette, inner="quartile", cut=0
        )
        if axes[i].get_legend() is not None: axes[i].get_legend().remove()
        
        # タイトル (効果量)
        axes[i].set_title(f"(d={d_values.get(col,0):.2f})", fontsize=FONT_SIZES['title'], pad=20)
        
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_xticks([0])
        
        # X軸ラベル
        axes[i].set_xticklabels([col], rotation=0, ha='center', fontsize=FONT_SIZES['xlabel'])
        axes[i].tick_params(axis='x', pad=20)
        axes[i].tick_params(axis='y', labelsize=FONT_SIZES['ylabel'])

        y_max = df_capped[col].max()
        axes[i].set_ylim(0, y_max)
        axes[i].yaxis.set_major_locator(LinearLocator(5))
        
        if y_max < 10:
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.tight_layout(pad=2.0)
    plt.savefig('RQ1 violine.pdf', bbox_inches='tight')
    print("Top3画像を保存しました: RQ1 violine.pdf")
else:
    print("条件を満たす特徴量がありません。")


# ---------------------------------------------------------
# 全特徴量プロット
# ---------------------------------------------------------
all_sorted_features = sorted(all_feature_cols, key=lambda x: d_values.get(x, 0), reverse=True)

print("\n" + "="*60)
print(f"全 {len(all_sorted_features)} 特徴量をプロットします")

cols_per_row = 5
total_plots = len(all_sorted_features)
rows = math.ceil(total_plots / cols_per_row)

fig_all, axes_all = plt.subplots(rows, cols_per_row, figsize=(40, 8 * rows))
axes_flat = axes_all.flatten()

for i, col in enumerate(all_sorted_features):
    ax = axes_flat[i]
    
    sns.violinplot(
        data=df, x=pd.Series(0, index=df.index), y=col, 
        hue=TARGET_COL, hue_order=order_list,
        split=True, ax=ax, palette=custom_palette, inner="quartile", cut=0
    )
    
    if ax.get_legend() is not None: ax.get_legend().remove()

    # タイトル
    ax.set_title(f"{col}\n(d={d_values.get(col, 0):.2f})", fontsize=30, pad=15)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=24)
    
    y_max = df[col].max()
    if pd.isna(y_max): y_max = 1
    
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(LinearLocator(5))
    
    if y_max < 5:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    elif y_max > 1000:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

plt.tight_layout(pad=3.0, h_pad=4.0)
output_file_all = 'RQ1 violine_all.pdf'
plt.savefig(output_file_all, bbox_inches='tight')
print(f"全メトリクスの画像を保存しました: {output_file_all}")