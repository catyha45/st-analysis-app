# -*- coding: utf-8 -*-

"""
林大神的實驗設計, DOE 的基本物件
Version 0.1 2026/05/09 我一定是瘋了…哈哈哈…
"""

import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import stats as sp_stats
import re
from itertools import combinations, product as iterproduct
import io


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


def center_point_curvature_test(y_factor, y_center):
    """中心點曲率 F 檢定
    y_factor : array-like, 因子點（非中心點）的 Y 數值
    y_center : array-like, 中心點的 Y 數值
    回傳包含檢定結果的 dict
    """
    import numpy as np
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
# 網頁基本設定與 Session State 初始化
# ==========================================
st.set_page_config(page_title="DoE 實驗數據分析與最佳化", layout="wide")
st.title("📈 實驗設計 (DoE) 分析與最佳化預測模型")
st.markdown("上傳數據、篩選顯著因子、驗證模型健康度，並自動尋找最佳生產參數。")

# 初始化 session state 來記憶最佳化參數
if 'opt_params' not in st.session_state:
    st.session_state.opt_params = {}
if "show_success_toast" not in st.session_state:
    st.session_state.show_success_toast = False
# 🌟 初始化計算狀態鎖
if "has_optimized_results" not in st.session_state:
    st.session_state.has_optimized_results = False

# ==========================================
# 步驟 1：檔案上傳與資料清理
# ==========================================
st.header("1. 上傳數據與資料總覽")
uploaded_file = st.file_uploader("📂 請上傳包含實驗數據的 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    cleaned_columns = [re.sub(r'\W+', '_', col).strip('_') for col in df.columns]
    if len(cleaned_columns) != len(set(cleaned_columns)):
        st.error("⚠️ 欄位名稱清理後出現重複，請修改 CSV 欄位名稱後重新上傳。")
        st.stop()
    df.columns = cleaned_columns
    
    with st.expander("👀 點擊查看原始資料與敘述統計", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df)
        with col2:
            st.dataframe(df.describe().transpose())

    st.divider()

    # ==========================================
    # 步驟 2：使用者設定 Y 與分析因子
    # ==========================================
    st.header("2. 設定模型參數")
    
    col_y, col_x = st.columns([1, 2])
    factors = list(df.columns)
    with col_y:
        y_col = st.selectbox("🎯 請選擇反應變數 (Y)", options=factors, index=len(factors) - 1)
    
    with col_x:
        available_x = [col for col in factors if col != y_col]
        x_cols = st.multiselect("⚙️ 請選擇要分析的因子 (X) [將自動產生二階交互作用]", options=available_x)

    # 若使用者改變了 X 因子，清除舊的最佳化記憶避免干擾
    if 'prev_x_cols' in st.session_state and st.session_state.prev_x_cols != x_cols:
        st.session_state.opt_params = {}
    st.session_state.prev_x_cols = x_cols

    if y_col and x_cols:
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            st.error(f"⚠️ 反應變數 `{y_col}` 必須是數值型欄位，請重新選擇。")
            st.stop()

        # 偵測數值因子與類別因子
        numeric_x = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
        categ_x   = [c for c in x_cols if not pd.api.types.is_numeric_dtype(df[c])]

        if categ_x:
            st.info(f"📌 已偵測到類別因子：**{', '.join(categ_x)}**，將自動套用虛擬變數編碼 C()。")

        st.divider()

        # ==========================================
        # 步驟 3：向後消去法 (Backward Elimination)
        # ==========================================
        st.header("3. OLS 模型自動篩選,「非」強制階層原則 (P < 0.1)")
        
        main_effects  = [to_term(c, categ_x) for c in x_cols]
        interactions  = [f"{to_term(a, categ_x)}:{to_term(b, categ_x)}" for a, b in combinations(x_cols, 2)]
        current_terms = main_effects + interactions

        model = None
        df_json = df.to_json()  # 序列化一次，供迴圈內快取使用
        with st.expander("🔄 展開查看模型自動優化過程", expanded=False):
            step = 1
            while True:
                if not current_terms:
                    st.warning("⚠️ 所有因子的 P-value 均大於 0.1，無法建立有效的預測模型。")
                    break

                formula = f"{y_col} ~ " + " + ".join(current_terms)
                model = fit_ols(formula, df_json)
                pvalues = model.pvalues.drop('Intercept', errors='ignore')

                # 以「整個 term」為單位彙總最大 p-value（處理類別因子多 level 的情況）
                term_max_p = {}
                for pname, pval in pvalues.items():
                    for term in current_terms:
                        if pval_belongs_to(pname, term):
                            term_max_p[term] = max(term_max_p.get(term, 0), pval)
                            break

                if not term_max_p:
                    break

                max_p_val  = max(term_max_p.values())
                max_p_term = max(term_max_p, key=term_max_p.get)

                if max_p_val > 0.1:
                    st.write(f"🔹 步驟 {step}: 移除 `{max_p_term}` (P-value = {max_p_val:.4f} > 0.1)")
                    current_terms.remove(max_p_term)
                    step += 1
                else:
                    st.success("✅ 篩選完成！保留的因子 P-value 均 <= 0.1。")
                    break
        
        if model is not None:
            # 提取係數並組合方程式字串
            params = model.params
            equation_parts = []
            
            for name, coef in params.items():
                if name == 'Intercept':
                    # 處理截距項
                    equation_parts.append(f"{coef:.4f}")
                else:
                    # 處理其餘因子，將 : 換成 * 並格式化正負號
                    display_name = name.replace(':', ' * ')
                    sign = " + " if coef >= 0 else " - "
                    equation_parts.append(f"{sign}{abs(coef):.4f} * {display_name}")
            
            # 將所有部分結合成完整方程式
            full_equation_text = f"{y_col} = " + "".join(equation_parts)

            # 顯示於 st.info
            st.info(f"**最終最佳化公式**:     \n\n`{full_equation_text}`")

            #st.text(model.summary())
            st.code(str(model.summary()), language="text")
            st.divider()

            # ==========================================
            # 步驟 3.5：中心點曲率分析（選配）
            # ==========================================
            st.header("3.5. 🟡 中心點曲率分析 (選配)")
            st.markdown("標記哪些實驗列為中心點，系統將自動執行 **曲率 F 檢定**，判斷一階線性模型是否足夠描述數據。")

            with st.expander("⚙️ 設定中心點，點擊展開", expanded=False):
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
                    all_indices = list(df.index)
                    selected_rows = st.multiselect(
                        f"選擇中心點列號（0 起始，共 {len(df)} 列）:",
                        options=all_indices,
                        key="center_rows"
                    )
                    if selected_rows:
                        center_mask = df.index.isin(selected_rows)

            # 執行檢定
            if 'center_mask' in dir() and center_mask is not None:
                n_c = int(center_mask.sum())
                n_f = int((~center_mask).sum())

                if n_c < 2:
                    st.warning("⚠️ 中心點至少需要 **2 筆**才能估計純誤差（目前僅 {n_c} 筆）。")
                elif n_f < 1:
                    st.warning("⚠️ 因子點數量不足，請檢查資料。")
                else:
                    y_factor_vals = df.loc[~center_mask, y_col].values
                    y_center_vals = df.loc[center_mask,  y_col].values
                    result = center_point_curvature_test(y_factor_vals, y_center_vals)

                    if result is None:
                        st.error("純誤差自由度不足，無法檢定。")
                    else:
                        # 顯示基本資訊
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("中心點均値",  f"{result['中心點均値']:.4f}")
                        c2.metric("因子點均値",  f"{result['因子點均値']:.4f}")
                        c3.metric("F 統計量",     f"{result['F 統計量']:.4f}")
                        c4.metric("P 値",         f"{result['P 値']:.4f}")

                        # 顯示完整檢定表
                        import pandas as pd
                        result_df = pd.DataFrame.from_dict(
                            {k: [v] for k, v in result.items() if k != "判斷"},
                            orient='columns'
                        )
                        st.dataframe(result_df, use_container_width=True, hide_index=True)

                        # 顯示判斷結果
                        if result['P 値'] < 0.05:
                            st.warning(f"📊 **{result['判斷']}**\n\n"
                                       "建議加入軸點，將實驗設計升階為 "
                                       "**Central Composite Design (CCD)** 或 **Box-Behnken Design**，"
                                       "並在模型中加入二次項 X²。")
                        else:
                            st.success(f"📊 **{result['判斷']}**\n\n"
                                       "目前線性模型（含二階交互作用）足以描述數據，不需要升階。")

            st.divider()

            # ==========================================
            # 步驟 4：殘差分析 (Residual Analysis)
            # ==========================================
            st.header("4. 📊 殘差分析 (驗證模型健康度)")
            st.markdown("良好的模型，其殘差應隨機散佈於 0 附近 (無明顯漏斗狀)，且 Q-Q 圖的點應貼近對角線。")
            
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.scatter(model.fittedvalues, model.resid, alpha=0.7, edgecolors='k')
                ax1.axhline(0, color='red', linestyle='--')
                ax1.set_xlabel("Fitted Values (預測值)")
                ax1.set_ylabel("Residuals (殘差)")
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

            st.divider()

            # ==========================================
            # 步驟 5：目標最佳化求解 (Optimization)
            # ==========================================
            st.header("5. 🎯 自動尋找最佳參數-因子範圍延伸 30%")
            st.markdown("設定您的目標，系統將利用演算法自動反推出最佳的實驗條件設定值。")
            
            col_opt1, col_opt2 = st.columns([1, 2])
            with col_opt1:
                opt_goal = st.selectbox("請選擇最佳化目標:", ["最大化 (Maximize)", "最小化 (Minimize)", "目標值 (Target)"])
                target_val = None
                if opt_goal == "目標值 (Target)":
                    target_val = st.number_input("請輸入期望的目標數值:", value=float(df[y_col].mean()))
            
            with col_opt2:
                st.write("") 
                st.write("") 
                start_opt = st.button("🚀 開始最佳化運算", type="primary", use_container_width=True)
            
            if start_opt:
                # 數值因子：設定搜尋範圍（延伸 30%）
                num_bounds = []
                for col in numeric_x:
                    span = df[col].max() - df[col].min()
                    num_bounds.append((df[col].min() - span * 0.3, df[col].max() + span * 0.3))

                # 類別因子：枚舉所有可能組合
                cat_levels = [df[col].unique().tolist() for col in categ_x]
                cat_combos = list(iterproduct(*cat_levels)) if categ_x else [()]

                best_res_fun   = float('inf')
                best_opt_x     = None
                best_opt_cats  = {}
                best_converged = False  # 防止未初始化導致 NameError

                with st.spinner("優化演算法多起點求解計算中（枚舉類別組合）..."):
                    for cat_combo in cat_combos:
                        fixed_cats = dict(zip(categ_x, cat_combo))

                        def objective(x_array, fixed_cats=fixed_cats):
                            row = {**fixed_cats, **dict(zip(numeric_x, x_array))}
                            pred_y = model.predict(pd.DataFrame([row]))[0]
                            if opt_goal == "最大化 (Maximize)": return -pred_y
                            elif opt_goal == "最小化 (Minimize)": return pred_y
                            else: return (pred_y - target_val) ** 2

                        if numeric_x:
                            res = opt.differential_evolution(objective, num_bounds, seed=42, tol=1e-8, maxiter=1000)
                            fun_val = res.fun
                            opt_x_vals = res.x
                            converged = res.success
                        else:
                            # 無數值因子：直接計算
                            fun_val = objective([])
                            opt_x_vals = []
                            converged = True

                        if fun_val < best_res_fun:
                            best_res_fun  = fun_val
                            best_opt_x    = opt_x_vals
                            best_opt_cats = fixed_cats
                            best_converged = converged

                if best_converged:
                    st.success("🎉 運算完成！已自動將最佳參數帶入下方的模擬器中。")

                    # 合併數值與類別最佳解
                    opt_params_combined = {**best_opt_cats,
                                           **dict(zip(numeric_x, best_opt_x if best_opt_x is not None else []))}

                    # 防止 rerun 之後消失
                    st.session_state.opt_params = opt_params_combined
                    st.session_state.has_optimized_results = True
                    st.session_state.show_success_toast = True

                    final_pred_df = pd.DataFrame([opt_params_combined])
                    opt_y = model.predict(final_pred_df)[0]

                    # 直接把最佳化結果，塞進輸入框要用的 key 裡面！
                    for col, best_val in opt_params_combined.items():
                        if col in categ_x:
                            st.session_state[f"select_{col}"] = best_val
                        else:
                            st.session_state[f"slider_{col}"] = float(best_val)

                    # 強制引發 Rerun，確保 Chrome 不失效
                    st.rerun()

                else:
                    st.error("⚠️ 演算法無法收斂，請檢查資料或放寬條件。")

            # 只要算過一次，Grid 就不會因為 Rerun 而消失
            if st.session_state.has_optimized_results:
    
                # 成功提示控制
                if st.session_state.show_success_toast:
                    st.success("🎉 運算完成！已自動將最佳參數帶入下方的模擬器中。")
                    st.session_state.show_success_toast = False

                # 重新預測
                current_opt_params = st.session_state.opt_params
                final_pred_df = pd.DataFrame([current_opt_params])
                opt_y = model.predict(final_pred_df)[0]

                # 重新預測
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
                    col_types = ["類別" if c in categ_x else "數值" for c in x_cols]
                    opt_df = pd.DataFrame({
                        "因子": x_cols,
                        "型別": col_types,
                        "最佳設定值": display_vals
                    }).set_index("因子")
        
                    st.dataframe(opt_df, use_container_width=True)            

            st.divider()

            # ==========================================
            # 步驟 6：互動式預測模擬器 (已串接記憶)
            # ==========================================
            st.header("6. 🎛️ 手動預測模擬器")
            st.markdown("調整下方參數以即時預測結果。若已執行最佳化運算，預設值會是**最佳參數**；類別因子以下拉選單選擇，數值因子以滑桿調整。")

            MAX_COLS_PER_ROW = 5  # 每行最多顯示 4 個因子，防止 UI 擠壓
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
                            user_inputs[col] = st.selectbox(
                                label=col,
                                options=options,
                                key=f"select_{col}"
                            )
                
                        else:

                            span = df[col].max() - df[col].min()
                            min_val = float(df[col].min() - span * 0.3)
                            max_val = float(df[col].max() + span * 0.3)
                
                            # 第一次開網頁，初始化這個 key 為平均值
                            if f"slider_{col}" not in st.session_state:
                                st.session_state[f"slider_{col}"] = float(df[col].mean())
                
                            # 如有最佳值 Slider 的 min/max 範圍，進行安全限制
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


            pred_df = pd.DataFrame([user_inputs])
            predicted_y = model.predict(pred_df)[0]

            st.metric(label=f"微調後的 {y_col} 預測數值", value=f"{predicted_y:.4f}", delta_color="off")