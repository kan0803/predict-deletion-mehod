import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from cliffs_delta import cliffs_delta  # ライブラリを使用
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

BINARY_PREFIXES = []
IGNORE_FOR_STATS_RANKING = ['is_truly_added']

outlier_caps = {
    'comment_loc_ratio': 0.4,
    'char_length': 6000,
    'code_loc': 160,
    'docstring_length': 500,
    'cyclomatic_complexity': 20,
    'halstead_volume': 1000,
    'call_expression_count': 20,
}

TOP_N = 3

FONT_SIZES = {
    'title': 55,
    'xlabel': 55,
    'ylabel': 55,
    'legend': 45
}

plt.rcdefaults()
plt.rcParams.update({'font.size': 40}) 

# ---------------------------------------------------------
# データ処理
# ---------------------------------------------------------
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"データを読み込みました: {CSV_FILE_PATH}")
except FileNotFoundError:
    print(f"エラー: {CSV_FILE_PATH} が見見つかりません。")
    exit()

all_feature_cols = [
    col for col in df.columns 
    if col in TARGET_FEATURES_LIST and pd.api.types.is_numeric_dtype(df[col])
]

# ---------------------------------------------------------
# 統計検定 & Top3選定 (Cliff's delta ライブラリ版)
# ---------------------------------------------------------
delta_values = {}
p_values = {}
groups = df[TARGET_COL].unique()

ranking_candidates = [
    c for c in all_feature_cols 
    if c not in IGNORE_FOR_STATS_RANKING and not any(c.startswith(pre) for pre in BINARY_PREFIXES)
]

if len(groups) == 2:
    group_a_label = groups[0]
    group_b_label = groups[1]
    group_a = df[df[TARGET_COL] == group_a_label]
    group_b = df[df[TARGET_COL] == group_b_label]

    print("\n" + "="*60)
    print("【統計検定結果 (Mann-Whitney U) & 効果量 (Cliff's delta)】")
    print("="*60)
    print(f"Group A: {group_a_label} (n={len(group_a)})")
    print(f"Group B: {group_b_label} (n={len(group_b)})\n")

    for col in all_feature_cols:
        # cliffs-delta ライブラリを使用
        d, _ = cliffs_delta(group_a[col], group_b[col])
        delta_values[col] = abs(d)
        
        # 検定
        _, p = mannwhitneyu(group_a[col], group_b[col], alternative='two-sided')
        p_values[col] = p
        
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        is_excluded = (col not in ranking_candidates)
        note = " (Top3対象外)" if is_excluded else ""

        # ラベル(medium等)の出力を削除
        print(f"{col:<25} | delta={abs(d):.3f} | p={p:.3e} {star}{note}")

    valid_candidates = [c for c in ranking_candidates if p_values[c] < 0.05]
    valid_candidates.sort(key=lambda x: delta_values[x], reverse=True)
    top_features = valid_candidates[:TOP_N]
else:
    print("ターゲットクラスが2つではありません。")
    top_features = []

# ---------------------------------------------------------
# Top 3 プロット
# ---------------------------------------------------------
if len(top_features) > 0:
    print("\n" + "="*60)
    print(f"Top {len(top_features)} 特徴量をプロットします: {top_features}")
    
    df_capped = df.copy()
    for col, limit in outlier_caps.items():
        if col in df_capped.columns:
            df_capped.loc[df_capped[col] > limit, col] = limit

    custom_palette = {'deleted': 'red', 'survived': 'C0'} 
    order_list = ['survived', 'deleted']

    fig, axes = plt.subplots(1, len(top_features), figsize=(30, 12))
    if len(top_features) == 1: axes = [axes]

    for i, col in enumerate(top_features):
        sns.violinplot(
            data=df_capped, x=pd.Series(0, index=df_capped.index), y=col,
            hue=TARGET_COL, hue_order=order_list,
            split=True, ax=axes[i], palette=custom_palette, inner="quartile", cut=0
        )
        if i == 0:
            handles, labels = axes[i].get_legend_handles_labels()
            labels = [l.capitalize() for l in labels]
            axes[i].legend(handles, labels, loc='upper left', fontsize=FONT_SIZES['legend'], 
                           frameon=True, facecolor='white', edgecolor='black')
        else:
            if axes[i].get_legend() is not None: axes[i].get_legend().remove()
        
        # タイトル
        axes[i].set_title(f"($\\delta$={delta_values.get(col,0):.2f})", fontsize=FONT_SIZES['title'], pad=20)
        axes[i].set_xlabel(""); axes[i].set_ylabel(""); axes[i].set_xticks([0])
        axes[i].set_xticklabels([col], rotation=0, ha='center', fontsize=FONT_SIZES['xlabel'])
        axes[i].tick_params(axis='x', pad=20); axes[i].tick_params(axis='y', labelsize=FONT_SIZES['ylabel'])

        y_max = df_capped[col].max() if not pd.isna(df_capped[col].max()) else 1
        axes[i].set_ylim(0, y_max)
        axes[i].yaxis.set_major_locator(LinearLocator(5))
        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f' if y_max < 10 else '%.0f'))

    plt.tight_layout(pad=2.0)
    plt.savefig('RQ1 violine.pdf', bbox_inches='tight')
    print("Top3画像を保存しました: RQ1 violine.pdf")

# ---------------------------------------------------------
# 全特徴量プロット (NameError: r_values を delta_values に修正)
# ---------------------------------------------------------
all_sorted_features = sorted(all_feature_cols, key=lambda x: delta_values.get(x, 0), reverse=True)
if all_sorted_features:
    print("\n" + "="*60)
    print(f"全 {len(all_sorted_features)} 特徴量をプロットします")
    cols_per_row = 5
    rows = math.ceil(len(all_sorted_features) / cols_per_row)
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
        ax.set_title(f"{col}\n($\\delta$={delta_values.get(col, 0):.2f})", fontsize=30, pad=15)
        
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=24)
        y_max = df[col].max() if not pd.isna(df[col].max()) else 1
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_locator(LinearLocator(5))
        
        if y_max < 5:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(pad=3.0, h_pad=4.0)
    plt.savefig('RQ1 violine_all.pdf', bbox_inches='tight')
    print("全メトリクスの画像を保存しました: RQ1 violine_all.pdf")