# -*- coding: utf-8 -*-

"""
Original DOE framework developed by Prof. Lin.George.

Enhanced and deployed by Neo and Cheng Terry.

林大神的實驗設計, DOE 的基本物件
Version 0.1 2026/05/09 我一定是瘋了…哈哈哈…
Version 0.2 2026/08/06
    1. 介面重構：輸入集中到側邊欄，主畫面以步驟導覽一次只顯示一個步驟，不再無限捲動
    2. 未上傳檔案時顯示資料格式要求與流程說明，取代原本的空白頁
    3. matplotlib 圖表一律不畫中文（軸標籤改用 F1/F2 代號 + 對照表），
       徹底避開雲端 Linux 無中文字型導致的方框 □
    4. 最佳化求解新增逐因子限制條件：數值因子可自訂上下限或鎖定，
       類別因子可排除不可用的選項，並在成果表標示觸界因子
    5. EPV 設定改用白話標籤與即時回饋，降低初學者的理解門檻
    6. 向後消去法改為回傳過程紀錄（backward_eliminate），與 UI 渲染解耦
    7.
"""


import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import stats as sp_stats
import re
import numpy as np
import seaborn as sns
from itertools import combinations, product as iterproduct
from patsy import dmatrix
import io

# 圖表一律不畫中文:字型是否存在依機器而異(雲端 Linux 沒有微軟正黑體),
# 與其偵測字型再祈禱裝得到,不如讓 matplotlib 只吐 ASCII —— 中文說明留在 Streamlit 元件裡,
# 由瀏覽器負責算繪,永遠不會變成方框 □。
APP_VERSION = "0.2 (2026/08/06)"  # 與檔頭 docstring 的版本紀錄同步

FACTOR_LABEL_PREFIX = "F"  # 熱力圖軸標籤代號,避免把中文欄名直接畫進圖裡
RESPONSE_LABEL = "Y"

MIN_RESID_DF = 3           # 殘差自由度下限：低於此值時 t 檢定與殘差圖失去判讀意義
P_VALUE_THRESHOLD = 0.05   # 保留因子的顯著水準門檻
DEFAULT_SAMPLES_PER_TERM = 10  # EPV 原則：每個模型參數預設需要 10 筆樣本支撐
HIGH_CORR_THRESHOLD = 0.8  # 判定因子高度共線的相關係數門檻
RANGE_EXTEND_RATIO = 0.3   # 最佳化與模擬器的因子範圍外推比例
MAX_COLS_PER_ROW = 5       # 模擬器每列最多顯示的因子數，防止 UI 擠壓


# unicode_minus 用的字元在 DejaVu Sans 也缺，關掉才不會讓負號變成方框
plt.rcParams['axes.unicode_minus'] = False


def make_plot_labels(numeric_x, y_col):
    """把因子欄名對應成 ASCII 代號，回傳 (欄名 → 代號 dict, 對照表 DataFrame)

    欄名可能含中文，直接當軸標籤會被 matplotlib 畫成方框，故以 F1/F2… 代號取代，
    真正的欄名改用 Streamlit 表格呈現。
    """
    label_map = {col: f"{FACTOR_LABEL_PREFIX}{i}" for i, col in enumerate(numeric_x, start=1)}
    label_map[y_col] = RESPONSE_LABEL

    legend_df = pd.DataFrame({
        "代號": list(label_map.values()),
        "欄位名稱": list(label_map.keys()),
        "角色": ["因子 (X)"] * len(numeric_x) + ["反應變數 (Y)"],
    })
    return label_map, legend_df


@st.cache_data
def fit_ols(formula, data_json):
    # 使用 io.StringIO 包裝以避免 Windows 平台因中文或長字串誤判為檔案路徑
    data = pd.read_json(io.StringIO(data_json))
    return smf.ols(formula=formula, data=data).fit()


def to_term(col, categ_x):
    """類別因子包 C()，數值因子直接使用欄位名稱"""
    return f"C({col})" if col in categ_x else col


def pval_belongs_to(pval_name, term):
    """判斷某個 p-value 名稱是否屬於某個 formula term"""
    return (pval_name == term
            or pval_name.startswith(term + "]")
            or pval_name.startswith(term + "["))


def count_design_cols(terms, data):
    """回傳 formula 展開後設計矩陣的欄數（含截距）
    直接用 patsy 展開，才能正確計入 C() 類別因子被拆成的多個 level 欄位
    """
    return dmatrix(" + ".join(terms), data=data, return_type="dataframe").shape[1]


def term_strength(term, y_col, data_json):
    """單項迴歸的調整後 R²，作為裁減時的刪除順序依據（越小越先刪）"""
    try:
        return fit_ols(f"{y_col} ~ {term}", data_json).rsquared_adj
    except Exception:
        # 無法單獨配適的項（如共線性）視為最弱，優先刪除
        return float("-inf")


def max_features_allowed(n_samples, samples_per_term):
    """依樣本數決定特徵（設計矩陣欄位，不含截距）數量上限

    同時受兩個條件約束，取較嚴格者：
      1. EPV 原則：每個參數至少要有 samples_per_term 筆樣本
      2. 自由度預算：必須留下 MIN_RESID_DF 個殘差自由度
    """
    by_epv = n_samples // samples_per_term
    by_resid_df = n_samples - MIN_RESID_DF - 1  # 扣掉截距
    return max(1, min(by_epv, by_resid_df))


def trim_terms_to_budget(terms, y_col, data, data_json, max_features):
    """把模型項裁減到 max_features 個設計矩陣欄位以內，回傳 (保留的項, 被刪除的項)

    小樣本 DoE 全展開二階交互作用極易飽和（df_resid <= 0），此時 p-value 全為 NaN，
    後向消去法會因 NaN 比較恆為 False 而誤判成「篩選完成」，故必須在建模前先卡預算。
    刪除順序：先砍交互作用中解釋力最弱者，主效應留到最後才動。
    """
    kept = list(terms)
    dropped = []

    # 扣 1 是因為 count_design_cols 的回傳值含截距，而上限是針對特徵數
    while kept and count_design_cols(kept, data) - 1 > max_features:
        candidates = [t for t in kept if ":" in t] or kept
        weakest = min(candidates, key=lambda t: term_strength(t, y_col, data_json))
        kept.remove(weakest)
        dropped.append(weakest)

    return kept, dropped


def backward_eliminate(terms, y_col, data, data_json, max_features):
    """向後消去法主迴圈，回傳 (最終模型, 保留的項, 過程紀錄)

    改為回傳過程紀錄而非直接 st.write，模型才能在任何步驟頁面下都先算完，
    紀錄則留給「模型篩選」頁面渲染 —— 側邊欄導覽一次只顯示一個步驟，
    若把輸出寫死在迴圈裡，切到其他頁面時模型就不存在了。
    """
    current_terms = list(terms)
    model = None
    log = []
    step = 1

    while True:
        if not current_terms:
            log.append(("warning", f"⚠️ 所有因子的 P-value 均大於 {P_VALUE_THRESHOLD}，無法建立有效的預測模型。"))
            break

        formula = f"{y_col} ~ " + " + ".join(current_terms)
        model = fit_ols(formula, data_json)
        pvalues = model.pvalues.drop('Intercept', errors='ignore')

        # 防呆：秩虧損時 p-value 為 NaN，而 NaN 的比較恆為 False，
        # 會讓下方判斷直接掉進 else 宣告「篩選完成」，故在此攔截並強制裁減
        if model.df_resid < MIN_RESID_DF or pvalues.isna().any():
            candidates = [t for t in current_terms if ":" in t] or current_terms
            weakest = min(candidates, key=lambda t: term_strength(t, y_col, data_json))
            log.append(("write", f"🔸 步驟 {step}: 模型秩虧損（無有效 P-value），強制移除 `{weakest}`"))
            current_terms.remove(weakest)
            step += 1
            continue

        # 以「整個 term」為單位彙總最大 p-value（處理類別因子多 level 的情況）
        term_max_p = {}
        for pname, pval in pvalues.items():
            for term in current_terms:
                if pval_belongs_to(pname, term):
                    term_max_p[term] = max(term_max_p.get(term, 0), pval)
                    break

        if not term_max_p:
            break

        max_p_val = max(term_max_p.values())
        max_p_term = max(term_max_p, key=term_max_p.get)
        n_features = count_design_cols(current_terms, data) - 1  # 不含截距

        if max_p_val > P_VALUE_THRESHOLD:
            log.append(("write", f"🔹 步驟 {step}: 移除 `{max_p_term}` "
                                 f"(P-value = {max_p_val:.4f} > {P_VALUE_THRESHOLD})"))
            current_terms.remove(max_p_term)
            step += 1
        elif n_features > max_features:
            # p 值全數達標但特徵仍超編時，繼續移除 p 值最大者直到符合上限
            log.append(("write", f"🔺 步驟 {step}: 特徵數 {n_features} > 上限 {max_features}，"
                                 f"移除 P-value 最大的 `{max_p_term}` (P-value = {max_p_val:.4f})"))
            current_terms.remove(max_p_term)
            step += 1
        else:
            log.append(("success", f"✅ 篩選完成！保留 {n_features} 個特徵（上限 {max_features}），"
                                   f"P-value 均 <= {P_VALUE_THRESHOLD}。"))
            break

    return model, current_terms, log


def build_equation_text(model, y_col):
    """把模型係數組成可讀的迴歸方程式字串"""
    equation_parts = []
    for name, coef in model.params.items():
        if name == 'Intercept':
            equation_parts.append(f"{coef:.4f}")
        else:
            # 將 : 換成 * 並把正負號拆到項首，避免出現 "+ -0.5" 這種寫法
            display_name = name.replace(':', ' * ')
            sign = " + " if coef >= 0 else " - "
            equation_parts.append(f"{sign}{abs(coef):.4f} * {display_name}")
    return f"{y_col} = " + "".join(equation_parts)


def factor_bounds(series):
    """回傳外推 RANGE_EXTEND_RATIO 後的因子搜尋範圍"""
    span = series.max() - series.min()
    return (float(series.min() - span * RANGE_EXTEND_RATIO),
            float(series.max() + span * RANGE_EXTEND_RATIO))


def center_point_curvature_test(y_factor, y_center):
    """中心點曲率 F 檢定
    y_factor : array-like, 因子點（非中心點）的 Y 數值
    y_center : array-like, 中心點的 Y 數值
    回傳包含檢定結果的 dict
    """
    n_F = len(y_factor)
    n_C = len(y_center)
    y_bar_F = np.mean(y_factor)
    y_bar_C = np.mean(y_center)

    ss_curvature = (n_F * n_C / (n_F + n_C)) * (y_bar_F - y_bar_C) ** 2
    ss_pe        = np.sum((np.array(y_center) - y_bar_C) ** 2)
    df_pe        = n_C - 1

    if df_pe < 1 or ss_pe == 0:
        return None  # 純誤差自由度不足

    ms_curvature = ss_curvature / 1
    ms_pe        = ss_pe / df_pe
    f_stat       = ms_curvature / ms_pe
    p_val        = 1 - sp_stats.f.cdf(f_stat, 1, df_pe)

    return {
        "n_因子點": n_F,
        "n_中心點": n_C,
        "因子點均値": round(y_bar_F, 4),
        "中心點均値": round(y_bar_C, 4),
        "SS_曲率": round(ss_curvature, 4),
        "SS_純誤差": round(ss_pe, 4),
        "F 統計量": round(f_stat, 4),
        "P 値": round(p_val, 4),
        "判斷": "⚠️ 曲率顯著 (建議升階為 RSM)" if p_val < 0.05 else "✅ 曲率不顯著 (線性模型適用)"
    }


# ==========================================
# 各步驟的渲染函式
# 一次只呼叫使用者選中的那一個,避免整頁塞滿七個步驟要一直捲
# ==========================================
def render_data_overview(df, y_col, x_cols, categ_x):
    st.header("1️⃣ 資料總覽")
    st.markdown("先確認資料讀進來的樣子正確：欄位沒跑掉、數值範圍合理、沒有異常的缺漏。")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("樣本數", f"{len(df)} 筆")
    col_b.metric("欄位數", f"{df.shape[1]} 欄")
    col_c.metric("缺失值", f"{int(df.isna().sum().sum())} 個")

    if df.isna().sum().sum() > 0:
        st.warning("⚠️ 資料中有缺失值，OLS 會自動剔除整列，可能造成設計不再正交。")

    if categ_x:
        st.info(f"📌 已偵測到類別因子：**{', '.join(categ_x)}**，將自動套用虛擬變數編碼 C()。")

    tab_raw, tab_desc = st.tabs(["原始資料", "敘述統計"])
    with tab_raw:
        st.dataframe(df, use_container_width=True)
    with tab_desc:
        st.dataframe(df.describe().transpose(), use_container_width=True)


def render_correlation(df, y_col, numeric_x):
    st.header("2️⃣ 因子相關性檢視")
    st.markdown(
        "建模前先檢查因子之間是否存在共線性。正交設計的 X 之間應接近 0，"
        "若出現高相關（|r| > 0.8），表示設計不正交或資料有缺漏，係數將難以解讀。"
    )
    st.caption("⚠️ 此圖僅供診斷，**不會**用來篩選因子 —— DoE 的因子是刻意設計要驗證的，"
               "主效應與 Y 單變數相關性低但交互作用顯著是常見情況。")

    # 僅取數值因子與 Y：類別因子無法直接計算相關係數
    corr_cols = numeric_x + [y_col]
    if len(corr_cols) < 2:
        st.info("數值因子不足 2 個，略過相關性熱力圖。")
        return

    # Spearman 秩相關對非線性關係與離群值較穩健
    corr = df[corr_cols].corr(method='spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # 遮罩上三角避免資訊重複

    # 軸標籤改用 ASCII 代號，圖裡不出現任何中文
    label_map, legend_df = make_plot_labels(numeric_x, y_col)
    corr_plot = corr.rename(index=label_map, columns=label_map)

    col_plot, col_legend = st.columns([3, 1])

    with col_plot:
        fig_corr, ax_corr = plt.subplots(figsize=(min(1.1 * len(corr_cols) + 2, 11),
                                                  min(0.9 * len(corr_cols) + 2, 9)))
        sns.heatmap(corr_plot, mask=mask, annot=True, annot_kws={'size': 8},
                    cmap='coolwarm', center=0, fmt='.2f', square=True,
                    linewidths=0.5, vmin=-1, vmax=1,
                    cbar_kws={'shrink': 0.8, 'label': 'Spearman rho'}, ax=ax_corr)
        ax_corr.set_title("Spearman Correlation Matrix", fontsize=13, pad=15)
        plt.setp(ax_corr.get_xticklabels(), rotation=0)
        plt.setp(ax_corr.get_yticklabels(), rotation=0)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    with col_legend:
        st.caption("代號對照")
        st.dataframe(legend_df, use_container_width=True, hide_index=True)

    # 主動點名高共線性的因子配對，使用者不必自己盯著方格找
    high_pairs = [
        f"`{a}` ↔ `{b}` (r = {corr.loc[a, b]:.2f})"
        for a, b in combinations(numeric_x, 2)
        if abs(corr.loc[a, b]) > HIGH_CORR_THRESHOLD
    ]
    if high_pairs:
        st.warning("⚠️ 偵測到高度共線的因子配對：\n\n- " + "\n- ".join(high_pairs)
                   + "\n\n建議擇一納入模型，或檢查實驗設計是否正交。")
    else:
        st.success(f"✅ 因子間無高度共線性（|r| 均 <= {HIGH_CORR_THRESHOLD}）。")


def render_model(df, y_col, model, dropped_terms, elimination_log, max_features, samples_per_term):
    st.header("3️⃣ 模型自動篩選")
    st.caption(f"向後消去法（Backward Elimination），「非」強制階層原則，門檻 P < {P_VALUE_THRESHOLD}")

    st.caption(
        f"共 {len(df)} 筆數據，設定為每個因子至少 {samples_per_term} 筆支撐 "
        f"→ 模型最多納入 **{max_features}** 個項目"
        f"（數據量上限 {len(df) // samples_per_term}、自由度上限 {len(df) - MIN_RESID_DF - 1}，取較嚴格者）"
    )

    if dropped_terms:
        st.warning(
            f"⚠️ 超出特徵數上限，已於建模前移除 **{len(dropped_terms)}** 個解釋力最弱的項："
            f"`{'`, `'.join(dropped_terms)}`\n\n"
            f"（{len(df)} 筆樣本至多容納 {max_features} 個特徵）"
        )

    if model is None:
        st.error("⚠️ 沒有任何因子通過顯著性檢定，無法建立模型。請放寬 EPV 或檢查資料。")
        return

    # 模型健康度先給三個數字，細節留在下方摘要
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("R²", f"{model.rsquared:.4f}")
    col_b.metric("調整後 R²", f"{model.rsquared_adj:.4f}")
    col_c.metric("殘差自由度", f"{int(model.df_resid)}")

    st.info(f"**最終最佳化公式**:     \n\n`{build_equation_text(model, y_col)}`")

    tab_coef, tab_log, tab_summary = st.tabs(["係數表", "篩選過程", "完整統計摘要"])

    with tab_coef:
        coef_table = pd.DataFrame({
            "係數": model.params.round(4),
            "標準誤": model.bse.round(4),
            "t 值": model.tvalues.round(4),
            "P 值": model.pvalues.round(4),
        })
        coef_table.index.name = "項目"
        st.dataframe(coef_table, use_container_width=True)

    with tab_log:
        for kind, message in elimination_log:
            getattr(st, kind)(message)

    with tab_summary:
        st.code(str(model.summary()), language="text")


def render_curvature(df, y_col, x_cols):
    st.header("4️⃣ 中心點曲率分析")
    st.caption("選配步驟 —— 只有實驗設計中安排了中心點才需要執行")
    st.markdown("標記哪些實驗列為中心點，系統將自動執行 **曲率 F 檢定**，判斷一階線性模型是否足夠描述數據。")

    center_mode = st.radio(
        "中心點標記方式:",
        ["不進行中心點分析", "使用旗標欄位 (列對應 1/True 為中心點)", "手動選擇列號"],
        key="center_mode"
    )

    center_mask = None
    if center_mode == "使用旗標欄位 (列對應 1/True 為中心點)":
        flag_candidates = [c for c in df.columns if c not in x_cols and c != y_col]
        if flag_candidates:
            flag_col = st.selectbox("選擇旗標欄位:", options=flag_candidates, key="flag_col")
            center_mask = df[flag_col].astype(bool)
        else:
            st.warning("⚠️ 未找到可用的旗標欄位（Y 與 X 以外的欄位）。")
    elif center_mode == "手動選擇列號":
        selected_rows = st.multiselect(
            f"選擇中心點列號（0 起始，共 {len(df)} 列）:",
            options=list(df.index),
            key="center_rows"
        )
        if selected_rows:
            center_mask = df.index.isin(selected_rows)

    if center_mask is None:
        return

    n_c = int(center_mask.sum())
    n_f = int((~center_mask).sum())

    if n_c < 2:
        st.warning(f"⚠️ 中心點至少需要 **2 筆**才能估計純誤差（目前僅 {n_c} 筆）。")
        return
    if n_f < 1:
        st.warning("⚠️ 因子點數量不足，請檢查資料。")
        return

    result = center_point_curvature_test(df.loc[~center_mask, y_col].values,
                                         df.loc[center_mask, y_col].values)
    if result is None:
        st.error("純誤差自由度不足，無法檢定。")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("中心點均値", f"{result['中心點均値']:.4f}")
    c2.metric("因子點均値", f"{result['因子點均値']:.4f}")
    c3.metric("F 統計量",   f"{result['F 統計量']:.4f}")
    c4.metric("P 値",       f"{result['P 値']:.4f}")

    result_df = pd.DataFrame.from_dict(
        {k: [v] for k, v in result.items() if k != "判斷"}, orient='columns'
    )
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    if result['P 値'] < 0.05:
        st.warning(f"📊 **{result['判斷']}**\n\n"
                   "建議加入軸點，將實驗設計升階為 "
                   "**Central Composite Design (CCD)** 或 **Box-Behnken Design**，"
                   "並在模型中加入二次項 X²。")
    else:
        st.success(f"📊 **{result['判斷']}**\n\n"
                   "目前線性模型（含二階交互作用）足以描述數據，不需要升階。")


def render_residuals(model):
    st.header("5️⃣ 殘差分析")
    st.markdown("良好的模型，其殘差應隨機散佈於 0 附近 (無明顯漏斗狀)，且 Q-Q 圖的點應貼近對角線。")

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(model.fittedvalues, model.resid, alpha=0.7, edgecolors='k')
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs. Fitted")
        ax1.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_plot2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sm.qqplot(model.resid, line='s', ax=ax2, alpha=0.7)
        ax2.set_title("Normal Q-Q Plot")
        ax2.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig2)
        plt.close(fig2)


def default_constraint_table(df, numeric_x):
    """建立數值因子的預設限制條件表（實驗範圍外推 RANGE_EXTEND_RATIO）"""
    rows = []
    for col in numeric_x:
        lower, upper = factor_bounds(df[col])
        rows.append({
            "因子": col,
            "下限": round(lower, 4),
            "上限": round(upper, 4),
            "實驗實際範圍": f"{df[col].min():g} ~ {df[col].max():g}",
        })
    return pd.DataFrame(rows)


def is_at_bound(value, bound, rel_tol=1e-3):
    """判斷最佳解是否貼在搜尋範圍的邊界上

    差異以範圍寬度normalize，因子單位不同（秒 vs. 公尺）時才有一致的判定標準。
    """
    if value is None:
        return False
    lower, upper = bound
    span = upper - lower
    if span == 0:
        return False
    return min(abs(value - lower), abs(value - upper)) / span < rel_tol


def constraint_status(col, value, bounds_map, fixed_map):
    """回傳因子在成果表中的限制狀態文字"""
    if col in fixed_map:
        return "🔒 已鎖定"
    if col not in bounds_map:
        return "—"  # 類別因子不適用數值邊界
    lower, upper = bounds_map[col]
    if is_at_bound(value, (lower, upper)):
        return "⚠️ 觸及下限" if abs(value - lower) < abs(value - upper) else "⚠️ 觸及上限"
    return "✅ 範圍內"


def render_constraint_editor(df, numeric_x, categ_x):
    """讓使用者逐一設定每個因子的可行範圍，回傳 (數值範圍 dict, 固定值 dict, 類別可選 level dict)

    製程上常有硬限制（機台上限、材料規格、某供應商停產），
    若不讓使用者設限，演算法會給出無法執行的「最佳解」而失去意義。
    數值因子的下限 == 上限 視為鎖定，該因子不進入搜尋空間直接當常數代入。
    """
    with st.expander("🔒 設定各因子的限制條件（可行範圍）", expanded=True):
        st.caption(
            f"預設為實驗範圍向外延伸 {int(RANGE_EXTEND_RATIO * 100)}%。"
            "請依製程實際能力調整 —— **把下限與上限設成相同數值即代表鎖定該因子**。"
        )

        bounds_map, fixed_map, allowed_levels = {}, {}, {}

        if numeric_x:
            st.markdown("**數值因子**")
            edited = st.data_editor(
                default_constraint_table(df, numeric_x),
                key="constraint_table",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "因子": st.column_config.TextColumn(disabled=True),
                    "下限": st.column_config.NumberColumn(format="%.4f"),
                    "上限": st.column_config.NumberColumn(format="%.4f"),
                    "實驗實際範圍": st.column_config.TextColumn(disabled=True,
                                                                help="原始數據的最小值 ~ 最大值"),
                },
            )

            for _, row in edited.iterrows():
                col, lower, upper = row["因子"], row["下限"], row["上限"]
                if pd.isna(lower) or pd.isna(upper):
                    st.error(f"⚠️ 因子 `{col}` 的上下限不可留空。")
                    return None
                if lower > upper:
                    st.error(f"⚠️ 因子 `{col}` 的下限 ({lower:g}) 大於上限 ({upper:g})，請修正。")
                    return None
                if lower == upper:
                    fixed_map[col] = float(lower)
                else:
                    bounds_map[col] = (float(lower), float(upper))

        if categ_x:
            st.markdown("**類別因子**（取消勾選代表該選項不可用）")
            cat_cols = st.columns(min(len(categ_x), 3))
            for i, col in enumerate(categ_x):
                with cat_cols[i % len(cat_cols)]:
                    levels = sorted(df[col].unique().tolist(), key=str)
                    chosen = st.multiselect(col, options=levels, default=levels,
                                            key=f"allowed_{col}")
                    if not chosen:
                        st.error(f"⚠️ 因子 `{col}` 至少要保留一個可用選項。")
                        return None
                    allowed_levels[col] = chosen

        # 摘要一行講清楚搜尋空間有多大，使用者才知道自己把範圍縮到什麼程度
        n_combos = int(np.prod([len(v) for v in allowed_levels.values()])) if allowed_levels else 1
        st.caption(f"➜ 搜尋空間：{len(bounds_map)} 個可調數值因子、"
                   f"{len(fixed_map)} 個已鎖定、{n_combos} 種類別組合")

    return bounds_map, fixed_map, allowed_levels


def render_optimizer(df, y_col, x_cols, numeric_x, categ_x, model):
    st.header("6️⃣ 自動尋找最佳參數")
    st.markdown("設定目標與各因子的可行範圍，系統將自動反推出最佳的實驗條件設定值。")

    constraints = render_constraint_editor(df, numeric_x, categ_x)
    if constraints is None:
        return  # 限制條件不合法，先讓使用者修正
    bounds_map, fixed_map, allowed_levels = constraints

    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        opt_goal = st.selectbox("請選擇最佳化目標:",
                                ["最大化 (Maximize)", "最小化 (Minimize)", "目標值 (Target)"])
        target_val = None
        if opt_goal == "目標值 (Target)":
            target_val = st.number_input("請輸入期望的目標數值:", value=float(df[y_col].mean()))

    with col_opt2:
        st.write("")
        st.write("")
        start_opt = st.button("🚀 開始最佳化運算", type="primary", use_container_width=True)

    if start_opt:
        # 只有未鎖定的數值因子進入搜尋空間，鎖定者以常數代入
        free_x = list(bounds_map.keys())
        num_bounds = [bounds_map[col] for col in free_x]

        # 類別因子：僅枚舉使用者允許的 level 組合
        cat_levels = [allowed_levels[col] for col in categ_x]
        cat_combos = list(iterproduct(*cat_levels)) if categ_x else [()]

        best_res_fun   = float('inf')
        best_opt_x     = None
        best_opt_cats  = {}
        best_converged = False  # 防止未初始化導致 NameError

        with st.spinner("優化演算法多起點求解計算中（枚舉類別組合）..."):
            for cat_combo in cat_combos:
                fixed_cats = dict(zip(categ_x, cat_combo))

                def objective(x_array, fixed_cats=fixed_cats):
                    row = {**fixed_cats, **fixed_map, **dict(zip(free_x, x_array))}
                    pred_y = model.predict(pd.DataFrame([row]))[0]
                    if opt_goal == "最大化 (Maximize)": return -pred_y
                    elif opt_goal == "最小化 (Minimize)": return pred_y
                    else: return (pred_y - target_val) ** 2

                if free_x:
                    res = opt.differential_evolution(objective, num_bounds, seed=42,
                                                     tol=1e-8, maxiter=1000)
                    fun_val, opt_x_vals, converged = res.fun, res.x, res.success
                else:
                    # 數值因子全被鎖定（或本來就沒有）：直接計算
                    fun_val, opt_x_vals, converged = objective([]), [], True

                if fun_val < best_res_fun:
                    best_res_fun   = fun_val
                    best_opt_x     = opt_x_vals
                    best_opt_cats  = fixed_cats
                    best_converged = converged

        if not best_converged:
            st.error("⚠️ 演算法無法收斂，請檢查資料或放寬限制條件。")
            return

        # 合併數值（含鎖定值）與類別最佳解
        opt_params_combined = {**best_opt_cats, **fixed_map,
                               **dict(zip(free_x, best_opt_x if best_opt_x is not None else []))}

        # 防止 rerun 之後消失
        st.session_state.opt_params = opt_params_combined
        st.session_state.has_optimized_results = True
        st.session_state.show_success_toast = True

        # 直接把最佳化結果，塞進輸入框要用的 key 裡面！
        for col, best_val in opt_params_combined.items():
            if col in categ_x:
                st.session_state[f"select_{col}"] = best_val
            else:
                st.session_state[f"slider_{col}"] = float(best_val)

        # 強制引發 Rerun，確保 Chrome 不失效
        st.rerun()

    # 只要算過一次，看板就不會因為 Rerun 而消失
    if not st.session_state.has_optimized_results:
        return

    if st.session_state.show_success_toast:
        st.success("🎉 運算完成！已自動將最佳參數帶入「手動模擬器」中。")
        st.session_state.show_success_toast = False

    current_opt_params = st.session_state.opt_params
    opt_y = model.predict(pd.DataFrame([current_opt_params]))[0]

    st.divider()
    st.write("### 🎯 最佳化成果綜合看板")
    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(label=f"最佳化預測結果 ({y_col})", value=f"{opt_y:.4f}")

    with col_res2:
        display_vals = [
            f"{current_opt_params[c]:.3f}"
            if isinstance(current_opt_params[c], (int, float))
            else str(current_opt_params[c])
            for c in x_cols
        ]
        opt_df = pd.DataFrame({
            "因子": x_cols,
            "型別": ["類別" if c in categ_x else "數值" for c in x_cols],
            "最佳設定值": display_vals,
            "限制狀態": [constraint_status(c, current_opt_params.get(c),
                                           bounds_map, fixed_map) for c in x_cols],
        }).set_index("因子")
        st.dataframe(opt_df, use_container_width=True)

    # 解落在邊界代表真正的最佳點可能在限制之外，這是使用者最該知道的一件事
    at_bound = [c for c in bounds_map if is_at_bound(current_opt_params.get(c), bounds_map[c])]
    if at_bound:
        st.warning(
            f"⚠️ 以下因子的最佳解剛好落在您設定的邊界上：`{'`, `'.join(at_bound)}`\n\n"
            "代表模型認為往邊界外走還會更好。若製程允許，可放寬該因子的限制再算一次；"
            "若不允許，則目前這組就是限制條件下的最佳解。"
        )


def render_simulator(df, y_col, x_cols, categ_x, model):
    st.header("7️⃣ 手動預測模擬器")
    st.markdown("調整下方參數以即時預測結果。若已執行最佳化運算，預設值會是**最佳參數**；"
                "類別因子以下拉選單選擇，數值因子以滑桿調整。")

    user_inputs = {}
    col_chunks = [x_cols[i:i + MAX_COLS_PER_ROW] for i in range(0, len(x_cols), MAX_COLS_PER_ROW)]

    for chunk in col_chunks:
        input_cols = st.columns(len(chunk))
        for j, col in enumerate(chunk):
            with input_cols[j]:
                if col in categ_x:
                    options = sorted(df[col].unique().tolist(), key=str)

                    # 第一次開網頁，初始化這個 key 為第一個選項
                    if f"select_{col}" not in st.session_state:
                        st.session_state[f"select_{col}"] = options[0]

                    # 使用 key 自動雙向綁定
                    user_inputs[col] = st.selectbox(label=col, options=options, key=f"select_{col}")
                else:
                    min_val, max_val = factor_bounds(df[col])

                    # 第一次開網頁，初始化這個 key 為平均值
                    if f"slider_{col}" not in st.session_state:
                        st.session_state[f"slider_{col}"] = float(df[col].mean())

                    # 如有最佳值超出 Slider 的 min/max 範圍，進行安全限制
                    current_slider_val = st.session_state[f"slider_{col}"]
                    if current_slider_val < min_val:
                        st.session_state[f"slider_{col}"] = min_val
                    elif current_slider_val > max_val:
                        st.session_state[f"slider_{col}"] = max_val

                    # 直接去讀取在 session_state 中的值
                    user_inputs[col] = st.slider(
                        label=col,
                        min_value=min_val,
                        max_value=max_val,
                        step=(max_val - min_val) / 100,
                        key=f"slider_{col}"
                    )

    predicted_y = model.predict(pd.DataFrame([user_inputs]))[0]
    st.metric(label=f"微調後的 {y_col} 預測數值", value=f"{predicted_y:.4f}", delta_color="off")


# ==========================================
# 網頁基本設定與 Session State 初始化
# ==========================================
st.set_page_config(page_title="DoE 實驗數據分析與最佳化", page_icon="📈", layout="wide")

# 初始化 session state 來記憶最佳化參數
if 'opt_params' not in st.session_state:
    st.session_state.opt_params = {}
if "show_success_toast" not in st.session_state:
    st.session_state.show_success_toast = False
# 🌟 初始化計算狀態鎖
if "has_optimized_results" not in st.session_state:
    st.session_state.has_optimized_results = False

# ==========================================
# 側邊欄：資料來源與模型設定
# 所有「輸入」集中在側邊欄,主畫面只放「輸出」,使用者不必在長頁面裡來回找設定
# ==========================================
with st.sidebar:
    st.title("📈 DoE 分析")

    # 掛名資訊獨立成資訊卡：原本擠成一行灰字，既不好讀也對不起原作者
    st.info(
        "**Original DOE Framework**  \n"
        "Prof. Lin.George\n\n"
        "**Enhanced & Deployed by**  \n"
        "Neo · Cheng Terry",
        icon="👥",
    )
    st.caption(f"Version {APP_VERSION}")

    st.divider()
    st.subheader("① 資料來源")
    uploaded_file = st.file_uploader("上傳實驗數據 CSV", type=["csv"])

if uploaded_file is None:
    # 尚未上傳時給明確的起手式,而不是一片空白讓人不知道要做什麼
    st.title("📈 實驗設計 (DoE) 分析與最佳化預測模型")
    st.markdown("上傳數據、篩選顯著因子、驗證模型健康度，並自動尋找最佳生產參數。")
    st.info("👈 請先從左側側邊欄上傳 CSV 檔案，接著選擇反應變數 (Y) 與要分析的因子 (X)。")
    st.markdown(
        """
        #### 資料格式要求
        | 要求 | 說明 |
        | --- | --- |
        | 每列一次實驗 | 一列 = 一個實驗條件組合的結果 |
        | Y 必須是數值 | 反應變數（如離型力、良率） |
        | X 可為數值或類別 | 類別因子會自動做虛擬變數編碼 |
        | 樣本數 | 至少 `MIN_RESID_DF + 2` = 5 筆，建議 12 筆以上 |

        #### 分析流程
        1. **資料總覽** — 確認資料讀取正確
        2. **因子相關性** — 檢查設計是否正交
        3. **模型自動篩選** — 向後消去法留下顯著因子
        4. **中心點曲率**（選配）— 判斷是否需要升階為 RSM
        5. **殘差分析** — 驗證模型健康度
        6. **最佳化求解** — 反推最佳製程參數
        7. **手動模擬器** — 微調參數即時預測
        """
    )
    st.stop()

df = pd.read_csv(uploaded_file)
cleaned_columns = [re.sub(r'\W+', '_', col).strip('_') for col in df.columns]
if len(cleaned_columns) != len(set(cleaned_columns)):
    st.error("⚠️ 欄位名稱清理後出現重複，請修改 CSV 欄位名稱後重新上傳。")
    st.stop()
df.columns = cleaned_columns

with st.sidebar:
    st.subheader("② 模型設定")
    factors = list(df.columns)
    y_col = st.selectbox("🎯 反應變數 (Y)", options=factors, index=len(factors) - 1)

    available_x = [col for col in factors if col != y_col]
    x_cols = st.multiselect("⚙️ 分析因子 (X)", options=available_x,
                            help="系統會自動加入所有兩兩交互作用項")

    # 若使用者改變了 X 因子，清除舊的最佳化記憶避免干擾
    if 'prev_x_cols' in st.session_state and st.session_state.prev_x_cols != x_cols:
        st.session_state.opt_params = {}
        st.session_state.has_optimized_results = False
    st.session_state.prev_x_cols = x_cols

    # EPV 比例開放調整：DoE 資料為正交設計、樣本刻意精簡，10:1 往往過嚴
    # 標籤刻意避開「EPV / 特徵」等術語，改用白話問句，初學者才知道自己在調什麼
    samples_per_term = st.number_input(
        "📏 每個因子至少要有幾筆數據支撐？",
        min_value=2, max_value=50, value=DEFAULT_SAMPLES_PER_TERM, step=1,
        help="統計上稱為 EPV (Events Per Variable)。\n\n"
             "數字**調高** → 模型只留最關鍵的少數因子，結論較可靠但可能漏掉真效應。\n\n"
             "數字**調低** → 模型納入較多因子，但小樣本下容易配適到雜訊（過度配適）。\n\n"
             "一般統計建議 10；DoE 是正交設計、樣本刻意精簡，調到 3～5 通常合理。"
    )
    # 把設定值直接翻成「會發生什麼事」，使用者不必自己換算
    st.caption(f"➜ 目前設定：模型最多納入 **{max_features_allowed(len(df), samples_per_term)}** 個項目"
               f"（{len(df)} 筆數據 ÷ {samples_per_term}）")

if not x_cols:
    st.title("📈 實驗設計 (DoE) 分析與最佳化預測模型")
    st.success(f"✅ 已讀取 **{len(df)}** 筆資料、**{df.shape[1]}** 個欄位。")
    st.info("👈 請在左側選擇至少一個分析因子 (X) 以開始建模。")
    st.dataframe(df, use_container_width=True)
    st.stop()

if not pd.api.types.is_numeric_dtype(df[y_col]):
    st.error(f"⚠️ 反應變數 `{y_col}` 必須是數值型欄位，請重新選擇。")
    st.stop()

# 樣本連一個主效應都撐不起時無法建模，直接擋下；否則使用者會拿到 R²=1 的飽和假模型
if len(df) - MIN_RESID_DF - 1 < 1:
    st.error(
        f"⚠️ 樣本數不足：目前僅 **{len(df)}** 筆資料，至少需要 {MIN_RESID_DF + 2} 筆"
        f"才能在保留 {MIN_RESID_DF} 個殘差自由度的前提下配適任何模型。"
    )
    st.stop()

# ==========================================
# 建模流程（不論停在哪個步驟頁面都先算完，後續步驟才有模型可用）
# ==========================================
# 偵測數值因子與類別因子
numeric_x = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
categ_x   = [c for c in x_cols if not pd.api.types.is_numeric_dtype(df[c])]

max_features = max_features_allowed(len(df), samples_per_term)
main_effects = [to_term(c, categ_x) for c in x_cols]
interactions = [f"{to_term(a, categ_x)}:{to_term(b, categ_x)}" for a, b in combinations(x_cols, 2)]

df_json = df.to_json()  # 序列化一次，供快取使用
with st.spinner("模型配適與因子篩選中..."):
    initial_terms, dropped_terms = trim_terms_to_budget(
        main_effects + interactions, y_col, df, df_json, max_features
    )
    model, final_terms, elimination_log = backward_eliminate(
        initial_terms, y_col, df, df_json, max_features
    )

# ==========================================
# 側邊欄：步驟導覽
# 用 radio 而非 st.tabs,因為最佳化會觸發 st.rerun(),tabs 會彈回第一頁
# ==========================================
STEP_DATA       = "1️⃣ 資料總覽"
STEP_CORR       = "2️⃣ 因子相關性"
STEP_MODEL      = "3️⃣ 模型篩選"
STEP_CURVATURE  = "4️⃣ 中心點曲率（選配）"
STEP_RESIDUAL   = "5️⃣ 殘差診斷"
STEP_OPTIMIZE   = "6️⃣ 最佳化求解"
STEP_SIMULATOR  = "7️⃣ 手動模擬器"

# 模型不存在時，後續依賴模型的步驟一律封鎖，避免點進去看到例外
MODEL_DEPENDENT = {STEP_MODEL, STEP_RESIDUAL, STEP_OPTIMIZE, STEP_SIMULATOR}
all_steps = [STEP_DATA, STEP_CORR, STEP_MODEL, STEP_CURVATURE,
             STEP_RESIDUAL, STEP_OPTIMIZE, STEP_SIMULATOR]
available_steps = all_steps if model is not None else [STEP_DATA, STEP_CORR, STEP_MODEL]

with st.sidebar:
    st.divider()
    st.subheader("③ 分析步驟")
    current_step = st.radio("跳至步驟", options=available_steps,
                            key="nav_step", label_visibility="collapsed")

    st.divider()
    if model is not None:
        st.metric("調整後 R²", f"{model.rsquared_adj:.4f}")
        st.caption(f"保留 {len(final_terms)} 個模型項 · 殘差自由度 {int(model.df_resid)}")
        if st.session_state.has_optimized_results:
            st.caption("✅ 已完成最佳化求解")
    else:
        st.caption("⚠️ 尚未建立有效模型")

# ==========================================
# 主畫面：只渲染選中的步驟
# ==========================================
if current_step == STEP_DATA:
    render_data_overview(df, y_col, x_cols, categ_x)
elif current_step == STEP_CORR:
    render_correlation(df, y_col, numeric_x)
elif current_step == STEP_MODEL:
    render_model(df, y_col, model, dropped_terms, elimination_log, max_features, samples_per_term)
elif current_step == STEP_CURVATURE:
    render_curvature(df, y_col, x_cols)
elif current_step == STEP_RESIDUAL:
    render_residuals(model)
elif current_step == STEP_OPTIMIZE:
    render_optimizer(df, y_col, x_cols, numeric_x, categ_x, model)
elif current_step == STEP_SIMULATOR:
    render_simulator(df, y_col, x_cols, categ_x, model)