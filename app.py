import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球投資組合分析系統", layout="wide", page_icon="📈")

# 設定中文字體
plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心計算函數 ---
def calculate_mdd(series):
    """計算最大回撤"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

# --- 3. 強化型數據抓取函數 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers_tw, tickers_us, start, end):
    data_dict = {}
    unique_tw = list(set(tickers_tw + ['0050']))
    unique_us = list(set(tickers_us + ['SPY']))
    
    for s in unique_tw:
        if not s: continue
        try:
            ticker = f"{s}.TW"
            yf_obj = yf.Ticker(ticker)
            df = yf_obj.history(start=start, end=end, interval="1d", auto_adjust=True)
            if not df.empty:
                data_dict[s] = df['Close']
        except:
            st.sidebar.warning(f"台股 {s} 抓取嘗試失敗")

    for s in unique_us:
        if not s: continue
        try:
            yf_obj = yf.Ticker(s)
            df = yf_obj.history(start=start, end=end, interval="1d", auto_adjust=True)
            if not df.empty:
                data_dict[s] = df['Close']
        except:
            st.sidebar.warning(f"美股 {s} 抓取嘗試失敗")
    return data_dict

# --- 4. 側邊欄 ---
with st.sidebar:
    st.header('🎯 標的設定')
    tw_in = st.text_input('台股代號', '1215,1419,2430,2891,9918')
    us_in = st.text_input('美股代號', 'DBC,GLD,SPY,VCIT,VNQ,VTV,VUG')
    
    st.header('📅 時間與資金')
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    initial_cap = st.number_input('本金', value=100000)
    rf_rate = st.number_input('無風險利率 (%)', value=4.0) / 100
    
    st.header('🎲 模擬設定')
    num_simulations = st.slider('蒙地卡羅次數', 1000, 5000, 2000)
    forecast_len = st.slider('預測天數', 30, 365, 180)

# --- 5. 主程式執行 ---

# 1. 初始化 Session State 狀態（防止拉桿觸發重新整理導致畫面消失）
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# 2. 點擊按鈕後，將狀態設為 True
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    st.session_state.analysis_started = True

# 3. 根據狀態決定是否顯示分析內容
if st.session_state.analysis_started:
    tw_list = [x.strip() for x in tw_in.split(',') if x.strip()]
    us_list = [x.strip().upper() for x in us_in.split(',') if x.strip()]
    
    with st.spinner('正在從 Yahoo Finance 節點抓取全球複權數據...'):
        raw_data = fetch_stock_data(tw_list, us_list, start_date, end_date)
        
        if not raw_data:
            st.error("❌ 所有來源均連線失敗。")
            st.stop()
            
        df_prices = pd.DataFrame(raw_data).ffill().dropna()
        returns = df_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    st.success(f"✅ 成功載入 {len(df_prices.columns)} 檔資產數據！")
    st.download_button("📥 下載調整後數據 (CSV)", df_prices.to_csv().encode('utf-8'), "data.csv")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 統計", "🔗 相關性", "💰 模擬", "📐 市場模型", "⚖️ 效率前緣", "🔮 預測", "🚨 (黑天鵝)壓力測試"])

    with tab1:
        st.subheader("📋 統計特徵")
        res_df = pd.DataFrame(index=returns.columns)
        total_days = (df_prices.index[-1] - df_prices.index[0]).days
        years = max(total_days / 365.25, 0.1) 
        
        res_df['年化報酬'] = (df_prices.iloc[-1] / df_prices.iloc[0]) ** (1 / years) - 1
        res_df['年化波動'] = returns.std() * np.sqrt(252)
        res_df['夏普比率'] = (res_df['年化報酬'] - rf_rate) / res_df['年化波動']
        res_df['最大回撤'] = [calculate_mdd(df_prices[c])[0] for c in df_prices.columns]
        
        res_df['符合常態'] = [("✅ 是" if stats.jarque_bera(returns[c])[1] > 0.05 else "❌ 否") for c in returns.columns]
        
        numeric_cols = ['年化報酬', '年化波動', '夏普比率', '最大回撤']
        st.dataframe(res_df.style.format({c: "{:.2%}" for c in numeric_cols}), use_container_width=True)
        
        cols = st.columns(2)
        for i, col in enumerate(returns.columns):
            with cols[i%2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(returns[col], bins=40, density=True, alpha=0.7, color='steelblue')
                ax.set_title(f"{col} Distribution of Returns")
                st.pyplot(fig)

    with tab2:
        st.subheader("🔗 相關性矩陣")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = returns.corr()
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

    with tab3:
        st.subheader("💰 財富累積曲線")
        st.line_chart((1 + returns).cumprod() * initial_cap)

    with tab4:
        st.subheader("📐 市場模型 (Beta)")
        beta_data = []
        for s in [c for c in returns.columns if c not in ['0050', 'SPY']]:
            if s.isdigit() and '0050' in returns.columns:
                mkt_ref = '0050'
            elif not s.isdigit() and 'SPY' in returns.columns:
                mkt_ref = 'SPY'
            else: continue
            common_df = pd.concat([returns[mkt_ref], returns[s]], axis=1).dropna()
            if len(common_df) > 10:
                slope, _, r_val, _, _ = stats.linregress(common_df.iloc[:,0], common_df.iloc[:,1])
                beta_data.append({"Asset": s, "Benchmark": mkt_ref, "Beta": slope, "R2": r_val**2})
        st.table(pd.DataFrame(beta_data))

# --- 在 tab5 之前先準備好最佳化所需的數據與函數 ---
    import scipy.optimize as sco
    import matplotlib.ticker as mtick # 引入百分比格式化工具
 
    # 1. 計算日均報酬與共變異矩陣
    mu = returns.mean()
    S = returns.cov()
 
    def get_portfolio_performance(weights, mu, S, rf_rate):
        # 計算年化報酬與年化波動
        p_ret = np.sum(mu * weights) * 252
        p_std = np.sqrt(np.dot(weights.T, np.dot(S * 252, weights)))
        return p_ret, p_std
 
    def neg_sharpe_ratio(weights, mu, S, rf_rate):
        p_ret, p_std = get_portfolio_performance(weights, mu, S, rf_rate)
        return -(p_ret - rf_rate) / p_std
 
    def minimize_volatility(weights, mu, S, rf_rate):
        return get_portfolio_performance(weights, mu, S, rf_rate)[1]
 
    with tab5:
        st.subheader("⚖️ 效率前緣與最佳配置 (Scipy Optimize)")
    
    # --- 1. 計算邏輯 (完全保留您的原始邏輯) ---
        num_assets = len(returns.columns)
        sim_res = np.zeros((3, num_simulations))
        for i in range(num_simulations):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            p_ret, p_std = get_portfolio_performance(w, mu, S, rf_rate)
            sim_res[0,i] = p_std
            sim_res[1,i] = p_ret
            sim_res[2,i] = (p_ret - rf_rate) / p_std 
 
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
 
        opt_sharpe = sco.minimize(neg_sharpe_ratio, init_guess, args=(mu, S, rf_rate), 
                                method='SLSQP', bounds=bounds, constraints=constraints)
        sharpe_ret, sharpe_vol = get_portfolio_performance(opt_sharpe.x, mu, S, rf_rate)
        best_weights = opt_sharpe.x 
 
        opt_vol = sco.minimize(minimize_volatility, init_guess, args=(mu, S, rf_rate), 
                            method='SLSQP', bounds=bounds, constraints=constraints)
        min_vol_ret, min_vol_vol = get_portfolio_performance(opt_vol.x, mu, S, rf_rate)
 
        target_returns = np.linspace(min_vol_ret, max(sharpe_ret, sim_res[1].max()) * 1.05, 50)
        frontier_vol = []
        for t_ret in target_returns:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: get_portfolio_performance(x, mu, S, rf_rate)[0] - t_ret})
            res = sco.minimize(minimize_volatility, init_guess, args=(mu, S, rf_rate), 
                            method='SLSQP', bounds=bounds, constraints=cons)
            frontier_vol.append(res.fun if res.success else np.nan)
 
        # --- 2. 上方區塊：效率前緣大圖 (單獨一排) ---
        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(sim_res[0,:], sim_res[1,:], c=sim_res[2,:], cmap='viridis', s=15, alpha=0.4, label='Random Portfolios')
        plt.colorbar(sc, label='Sharpe Ratio')
        ax.plot(frontier_vol, target_returns, 'b-', linewidth=2.5, label='Efficient Frontier')
    
        asset_ret = mu * 252
        asset_vol = np.sqrt(np.diag(S)) * np.sqrt(252)
        ax.scatter(asset_vol, asset_ret, marker='o', color='grey', s=50, label='Individual Assets')
        for i, txt in enumerate(returns.columns):
            ax.annotate(txt, (asset_vol[i], asset_ret[i]), xytext=(5,0), textcoords='offset points')
 
        ax.scatter(min_vol_vol, min_vol_ret, marker='*', color='orange', s=250, edgecolors='black', label='Min Volatility (MVP)', zorder=10)
        ax.scatter(sharpe_vol, sharpe_ret, marker='*', color='purple', s=250, edgecolors='black', label='Max Sharpe (MSR)', zorder=10)
    
        cml_x = np.linspace(0, max(sim_res[0].max(), sharpe_vol)*1.2, 100)
        cml_slope = (sharpe_ret - rf_rate) / sharpe_vol
        ax.plot(cml_x, rf_rate + cml_slope * cml_x, 'g--', label='Capital Market Line (CML)', alpha=0.7)
 
        ax.set_title(f"Efficient Frontier & Optimal Portfolios (Rf={rf_rate*100:.2f}%)", fontsize=14)
        ax.set_xlabel("Annualized Volatility (Risk)")
        ax.set_ylabel("Annualized Expected Return")
        ax.legend(loc='best')
        st.pyplot(fig)
 
        st.markdown("---")
 
    # --- 3. 下方區塊：兩個圓餅圖並排 ---
        col_left, col_right = st.columns(2)
 
        with col_left:
            st.write("#### 🏆 Maximum Sharpe Ratio (MSR)")
            df_sharpe = pd.DataFrame({'Asset': returns.columns, 'Weight': best_weights * 100})
            df_sharpe = df_sharpe.sort_values(by='Weight', ascending=False)
            fig_pie1, ax_pie1 = plt.subplots(figsize=(4, 4))
            ax_pie1.pie(df_sharpe['Weight'], labels=df_sharpe['Asset'], autopct='%1.1f%%', startangle=90)
            st.pyplot(fig_pie1)
            st.dataframe(df_sharpe.style.format({'Weight': '{:.2f}%'}), hide_index=True, use_container_width=True)
            st.info(f"Ret: {sharpe_ret:.2%} / Vol: {sharpe_vol:.2%}")
 
        with col_right:
            st.write("#### 🛡️ Minimum Variance Portfolio (MVP)")
            df_mvp = pd.DataFrame({'Asset': returns.columns, 'Weight': opt_vol.x * 100})
            df_mvp = df_mvp.sort_values(by='Weight', ascending=False)
            fig_pie2, ax_pie2 = plt.subplots(figsize=(4, 4))
            ax_pie2.pie(df_mvp['Weight'], labels=df_mvp['Asset'], autopct='%1.1f%%', startangle=90)
            st.pyplot(fig_pie2)
            st.dataframe(df_mvp.style.format({'Weight': '{:.2f}%'}), hide_index=True, use_container_width=True)
            st.info(f"Ret: {min_vol_ret:.2%} / Vol: {min_vol_vol:.2%}")            


# --- TAB 6: 完全修復版 (恢復完整統計指標 + 參數看板 + 雙重分析) ---
    with tab6:
        st.markdown("#### 🔮 蒙地卡羅預測：幾何布朗運動模型 (GBM Simulation)")

        # 1. 參數設定
        n_sim_total = 1000  # 模擬次數
        n_plot = 50         # 繪圖路徑數
        
        # 2. 核心參數準備 (MSR)
        port_returns_series = (returns * best_weights).sum(axis=1)
        
        # 計算真正的年化參數 (顯示於看板)
        mu_p = port_returns_series.mean() * 252
        sigma_p = port_returns_series.std() * np.sqrt(252)
        
        s0 = initial_cap
        dt = 1/252
        
        # --- 參數看板 (顯示真正的模型輸入值) ---
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 5px solid #0984e3; margin-bottom: 20px;">
            <h5 style="margin:0; color: #2d3436;">⚙️ 模型輸入參數 (Model Input Parameters)</h5>
            <p style="margin:5px 0 0 0; color: #636e72; font-size: 14px;">
                此模擬基於 <b>最大夏普比率組合 (MSR)</b> 之歷史統計特徵：<br>
                • <b>年化預期報酬率 ($\mu$)</b> : <span style="color: #d63031; font-weight: bold;">{mu_p:.2%}</span><br>
                • <b>年化波動率 ($\sigma$)</b> : <span style="color: #d63031; font-weight: bold;">{sigma_p:.2%}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. 執行 GBM 模擬
        sim_paths = np.zeros((forecast_len, n_sim_total))
        sim_paths[0] = s0
        
        drift = (mu_p - 0.5 * sigma_p**2) * dt
        shock = sigma_p * np.sqrt(dt)
        
        z_matrix = np.random.normal(0, 1, (forecast_len - 1, n_sim_total))
        
        for t in range(1, forecast_len):
            sim_paths[t] = sim_paths[t-1] * np.exp(drift + shock * z_matrix[t-1])
            
        # 4. 路徑模擬圖
        st.write(f"**📈 資產價格路徑模擬 (Asset Price Paths)**")
        st.line_chart(sim_paths[:, :n_plot])

        # 5. 數據前處理
        final_values = sim_paths[-1, :]
        daily_diff = np.diff(sim_paths, axis=0)
        up_counts = np.sum(daily_diff > 0, axis=0)

        # ==========================================
        # Part A: 連續變數分析 (Terminal Wealth)
        # ==========================================
        st.markdown("### 1 連續變數分析：期末財富分佈 (Terminal Wealth)")
        
        # --- 這裡恢復您原本完整的統計計算 ---
        # 1. 基礎統計
        mean_end = np.mean(final_values)
        median_end = np.median(final_values)
        
        # 2. 獲利能力指標
        max_profit = np.max(final_values) - s0 # 最大獲利
        prob_profit = np.sum(final_values > s0) / n_sim_total # 勝率
        
        # 3. 風險指標 (從模擬結果反推)
        log_returns = np.log(final_values / s0)
        realized_vol = np.std(log_returns) / np.sqrt(forecast_len / 252) # 實現波動率
        
        # 4. 風險溢酬
        rf_end_value = s0 * np.exp(rf_rate * (forecast_len / 252))
        risk_premium = mean_end - rf_end_value
        
        # 5. 下檔風險
        var_95 = np.percentile(final_values, 5)
        cvar_95 = final_values[final_values <= var_95].mean()
        
        # --- 顯示完整的統計看板 (3欄位) ---
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1: 
            st.metric("平均期末資產", f"${mean_end:,.0f}", delta=f"{(mean_end/s0 -1):.2%}")
            st.metric("中位數資產", f"${median_end:,.0f}")
            st.metric("模擬年化波動率", f"{realized_vol:.2%}", help="從模擬路徑反推的實際波動程度")

        with col_c2: 
            st.metric("正報酬機率 (Win Rate)", f"{prob_profit:.1%}")
            st.metric("預期最大獲利 (Net)", f"${max_profit:,.0f}")
            st.metric("預期風險溢酬", f"${risk_premium:,.0f}", help=f"平均終值 - 無風險利率終值 (${rf_end_value:,.0f})")
            
        with col_c3: 
            st.markdown("#### ⚠️ 下檔風險 (Tail Risk)")
            st.metric("風險值 VaR (95%)", f"${var_95:,.0f}", delta=f"{(var_95/s0 -1):.2%}", delta_color="inverse")
            st.caption(f"條件風險值 CVaR (最差5%平均): ${cvar_95:,.0f}")

        # 擬合分析
        dist_candidates_cont = {
            "Log-Normal": stats.lognorm,
            "Gamma": stats.gamma,
            "Student's t": stats.t,
            "Chi-Squared": stats.chi2,
            "Beta": stats.beta
        }
        
        fit_results_cont = []
        for name, dist in dist_candidates_cont.items():
            try:
                params = dist.fit(final_values)
                D, p = stats.kstest(final_values, dist.cdf, args=params)
                fit_results_cont.append({"Model": name, "D_Statistic": D, "params": params, "dist": dist})
            except: pass
        
        fit_results_cont.sort(key=lambda x: x['D_Statistic'])
        best_fit_cont = fit_results_cont[0]

        # 繪圖 A (英文圖表)
        col_plot_c, col_rank_c = st.columns([3, 1])
        with col_plot_c:
            fig_cont, ax_cont = plt.subplots(figsize=(10, 5))
            ax_cont.hist(final_values, bins=60, density=True, alpha=0.5, color='#3498db', label='Simulated Data', edgecolor='white')
            x_fit = np.linspace(np.min(final_values), np.max(final_values), 200)
            pdf_fit = best_fit_cont['dist'].pdf(x_fit, *best_fit_cont['params'])
            ax_cont.plot(x_fit, pdf_fit, 'r-', lw=2.5, label=f"Best Fit: {best_fit_cont['Model']}")
            
            # English Labels
            ax_cont.set_title(f"Continuous Fit: Terminal Wealth (Best: {best_fit_cont['Model']})", fontsize=12)
            ax_cont.set_xlabel("Portfolio Value ($)", fontsize=10)
            ax_cont.set_ylabel("Probability Density", fontsize=10)
            ax_cont.legend(loc='upper right')
            import matplotlib.ticker as mticker
            ax_cont.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
            st.pyplot(fig_cont)
            
        with col_rank_c:
            st.markdown("**擬合優度 (Goodness of Fit)**")
            st.dataframe(pd.DataFrame(fit_results_cont)[['Model', 'D_Statistic']], hide_index=True)

        # A部分：統計解釋
        winner_model_c = best_fit_cont['Model']
        explanation_c = ""
        if "Log-Normal" in winner_model_c:
            explanation_c = "符合 **幾何布朗運動 (GBM)** 的理論預期。資產價格因具有複利效應且恆為正值，故呈現右偏的對數常態分佈。"
        elif "Student's t" in winner_model_c:
            explanation_c = "分佈具有**厚尾 (Fat Tails)** 特徵。這意味著市場出現極端黑天鵝事件的機率，比標準常態分佈預測的還要高。"
        else:
            explanation_c = "數據分佈呈現特定統計特徵，可能受短期波動或參數設定影響。"

        st.info(f"💡 **統計結論**：\n最佳擬合模型為 **{winner_model_c}**。\n\n**📝 專業解讀**：{explanation_c}")

        st.markdown("---")

        # ==========================================
        # Part B: 離散變數分析 (Up Days Frequency)
        # ==========================================
        st.markdown("### 2 離散變數分析：正報酬頻率分佈 (Positive Return Frequency)")
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; font-size: 13px; color: #333; border: 1px solid #ddd;">
            <b>📊 統計理論基礎 (Theoretical Basis)：</b><br>
            基於 <b>隨機漫步假說 (Random Walk Hypothesis)</b>，資產每日價格變動可近似為一系列獨立的 <b>伯努利試驗 (Bernoulli Trials)</b>。<br>
            本節分析預測期間內，資產呈現<b>正報酬 (Positive Return)</b> 交易日之計數分佈。<br>
            理論上，若市場為隨機漫步，其分佈應收斂於 <b>二項分佈 (Binomial Distribution)</b>。
        </div>
        """, unsafe_allow_html=True)

        # 統計指標
        mean_up = np.mean(up_counts)
        prob_up = mean_up / forecast_len
        std_up = np.std(up_counts)
        
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1: st.metric("期望正報酬天數", f"{mean_up:.1f} Days")
        with col_d2: st.metric("單日正報酬機率 $p$", f"{prob_up:.2%}")
        with col_d3: st.metric("頻率標準差", f"{std_up:.2f}")

        # 擬合分析
        x_min, x_max = np.min(up_counts), np.max(up_counts)
        x_discrete = np.arange(x_min, x_max + 1)
        observed_counts = np.bincount(up_counts.astype(int))
        if len(observed_counts) > x_max:
            observed_pmf = observed_counts[x_min : x_max+1] / n_sim_total
        else:
            observed_pmf = np.zeros(len(x_discrete))
            
        discrete_candidates = []
        
        # (A) Binomial
        binom_pmf = stats.binom.pmf(x_discrete, n=forecast_len, p=prob_up)
        rmse_binom = np.sqrt(np.mean((observed_pmf - binom_pmf)**2))
        discrete_candidates.append({"Model": "Binomial Dist.", "RMSE": rmse_binom, "pmf": binom_pmf})

        # (B) Poisson
        poisson_pmf = stats.poisson.pmf(x_discrete, mu=mean_up)
        rmse_poisson = np.sqrt(np.mean((observed_pmf - poisson_pmf)**2))
        discrete_candidates.append({"Model": "Poisson Dist.", "RMSE": rmse_poisson, "pmf": poisson_pmf})

        # (C) Normal Approx
        norm_approx_pdf = stats.norm.pdf(x_discrete, loc=mean_up, scale=std_up)
        norm_approx_pmf = norm_approx_pdf / np.sum(norm_approx_pdf)
        rmse_norm = np.sqrt(np.mean((observed_pmf - norm_approx_pmf)**2))
        discrete_candidates.append({"Model": "Normal Approx.", "RMSE": rmse_norm, "pmf": norm_approx_pmf})

        discrete_candidates.sort(key=lambda x: x['RMSE'])
        best_discrete = discrete_candidates[0]

        # 繪圖 B (英文圖表)
        col_plot_d, col_rank_d = st.columns([3, 1])
        with col_plot_d:
            fig_disc, ax_disc = plt.subplots(figsize=(10, 5))
            # Bar
            ax_disc.bar(x_discrete, observed_pmf, alpha=0.6, color='#f39c12', label='Observed Frequency', zorder=1)
            # Line (Best)
            ax_disc.plot(x_discrete, best_discrete['pmf'], 'b-o', ms=5, lw=2.5, label=f"Best Fit: {best_discrete['Model']}", zorder=2)
            # Binomial Reference
            if "Binomial" not in best_discrete['Model']:
                binom_res = next(item for item in discrete_candidates if "Binomial" in item["Model"])
                ax_disc.plot(x_discrete, binom_res['pmf'], 'k--', alpha=0.5, lw=1.5, label='Theoretical: Binomial', zorder=2)

            # English Labels
            ax_disc.set_title(f"Discrete Fit: Frequency of Positive Returns (Winner: {best_discrete['Model']})", fontsize=12)
            ax_disc.set_xlabel("Number of Positive Days", fontsize=10)
            ax_disc.set_ylabel("Probability Mass", fontsize=10)
            ax_disc.legend(loc='upper left')
            st.pyplot(fig_disc)
            
        with col_rank_d:
            st.markdown("**模型擬合排名**")
            st.caption("均方根誤差 RMSE (越低越佳)")
            for res in discrete_candidates:
                st.markdown(f"**{res['Model']}**")
                st.caption(f"RMSE: {res['RMSE']:.4f}")
        
        # B部分：統計解釋
        winner_model_d = best_discrete['Model']
        explanation_d = ""
        if "Binomial" in winner_model_d:
            explanation_d = "完全符合 **隨機漫步假說 (Random Walk)**。這證實了每日的價格漲跌可被視為獨立的事件，且符合理論上的二項分佈特徵。"
        elif "Normal Approx" in winner_model_d:
            explanation_d = "體現 **中央極限定理 (Central Limit Theorem)**。當預測天數 (N) 足夠大時，二項分佈會自然收斂至常態分佈，這是大樣本下的正常統計現象。"
        else:
            explanation_d = "模擬結果顯示出特殊的計數分佈特徵。"

        st.info(f"💡 **統計結論**：\n最佳擬合模型為 **{winner_model_d}**。\n\n**📝 專業解讀**：{explanation_d}")

        # --- TAB 7: 壓力測試 ---
        with tab7:
            st.subheader("🚨 投資組合壓力測試 (Stress Test)")
            
            # 1. 計算組合的加權 Beta (反映組合對市場的敏感度)
            # 這裡從你 TAB 4 的 beta_data 提取資料
            if len(beta_data) > 0:
                df_beta = pd.DataFrame(beta_data)
                # 建立權重字典方便查詢
                weight_dict = dict(zip(returns.columns, best_weights))
                # 計算組合 Beta = Σ (權重 * 個股 Beta)
                df_beta['Weighted Beta'] = df_beta.apply(lambda x: x['Beta'] * weight_dict.get(x['Asset'], 0), axis=1)
                port_beta = df_beta['Weighted Beta'].sum()
            else:
                port_beta = 1.0 # 預設值
                
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write("**自定義市場衝擊預測**")
                mkt_shock = st.slider("假設大盤(市場基準)下跌 (%)", -50, 0, -10)
                
                # 預估損失 = 本金 * 市場跌幅 * 組合 Beta
                est_loss_pct = (mkt_shock / 100) * port_beta
                est_loss_amt = initial_cap * est_loss_pct
                
                st.metric("預估組合跌幅", f"{est_loss_pct:.2%}", delta=f"{est_loss_pct:.2%}")
                st.metric("預估損失金額", f"${est_loss_amt:,.0f}")
                
            with col2:
                st.write("**歷史極端情境模擬**")
                scenarios = {
                    "2008 金融海嘯 (假設大盤 -20%)": -0.20,
                    "2020 疫情崩盤 (假設大盤 -15%)": -0.15,
                    "2022 升息縮表 (假設大盤 -10%)": -0.10,
                    "微幅修正 (假設大盤 -5%)": -0.05
                }
                
                scene_data = []
                for name, shock in scenarios.items():
                    loss_pct = shock * port_beta
                    scene_data.append({
                        "情境": name,
                        "大盤跌幅": f"{shock:.0%}",
                        "組合預估跌幅": f"{loss_pct:.2%}",
                        "預估損失金額": f"${initial_cap * loss_pct:,.0f}"
                    })
                
                st.table(pd.DataFrame(scene_data))
    
            st.info(f"💡 註：目前組合的加權 Beta 為 **{port_beta:.2f}**。這代表當大盤下跌 1% 時，預計你的組合會隨之變動 {abs(port_beta):.2f}%。")
