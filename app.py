import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import scipy.optimize as sco  # 引入最佳化套件
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球投資組合分析系統 (Pro)", layout="wide", page_icon="📈")

# 設定中文字體 (相容 Windows/Mac)
plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'Heiti TC', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心計算函數 ---
def calculate_mdd(series):
    """計算最大回撤"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

def get_portfolio_performance(weights, mean_returns, cov_matrix, rf):
    """計算組合回報與風險"""
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
    """最小化負夏普比率 (即最大化夏普)"""
    p_ret, p_var = get_portfolio_performance(weights, mean_returns, cov_matrix, rf)
    return -(p_ret - rf) / p_var

def minimize_volatility(weights, mean_returns, cov_matrix, rf):
    """最小化波動率"""
    p_ret, p_var = get_portfolio_performance(weights, mean_returns, cov_matrix, rf)
    return p_var

# --- 3. 數據抓取函數 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers_tw, tickers_us, start, end):
    data_dict = {}
    unique_tw = list(set(tickers_tw + ['0050']))
    unique_us = list(set(tickers_us + ['SPY']))
    
    # 處理台股
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

    # 處理美股
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
    tw_in = st.text_input('台股代號', '2330,2454,2317,2891,1215')
    us_in = st.text_input('美股代號', 'NVDA,AAPL,MSFT,TLT,GLD')
    
    st.header('📅 時間與資金')
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    initial_cap = st.number_input('本金', value=1000000)
    rf_rate = st.number_input('無風險利率 (%)', value=3.5) / 100
    
    st.header('🎲 模擬設定')
    num_simulations = st.slider('蒙地卡羅背景點數', 500, 3000, 1000)
    forecast_len = st.slider('預測天數', 30, 365, 180)

# --- 5. 主程式執行 ---

if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    st.session_state.analysis_started = True

if st.session_state.analysis_started:
    tw_list = [x.strip() for x in tw_in.split(',') if x.strip()]
    us_list = [x.strip().upper() for x in us_in.split(',') if x.strip()]
    
    with st.spinner('正在從 Yahoo Finance 運算全球複權數據...'):
        raw_data = fetch_stock_data(tw_list, us_list, start_date, end_date)
        
        if not raw_data:
            st.error("❌ 所有來源均連線失敗。")
            st.stop()
            
        df_prices = pd.DataFrame(raw_data).ffill().dropna()
        if len(df_prices.columns) < 2:
            st.error("❌ 資產數量不足，請至少輸入兩檔有效標的以進行組合分析。")
            st.stop()
            
        returns = df_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    st.success(f"✅ 成功載入 {len(df_prices.columns)} 檔資產數據！")
    
    # 計算基礎統計量 (日資料)
    mu = returns.mean() 
    S = returns.cov() 

    # 用於後續計算的最佳權重 (先初始化)
    best_weights_global = np.array([1/len(returns.columns)] * len(returns.columns))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 統計", "🔗 相關性", "💰 模擬", "📐 市場模型", "⚖️ 效率前緣(Pro)", "🔮 預測", "🚨 壓力測試"])

    with tab1:
        st.subheader("📋 統計特徵")
        res_df = pd.DataFrame(index=returns.columns)
        total_days = (df_prices.index[-1] - df_prices.index[0]).days
        years = max(total_days / 365.25, 0.1) 
        
        res_df['年化報酬'] = (df_prices.iloc[-1] / df_prices.iloc[0]) ** (1 / years) - 1
        res_df['年化波動'] = returns.std() * np.sqrt(252)
        res_df['夏普比率'] = (res_df['年化報酬'] - rf_rate) / res_df['年化波動']
        res_df['最大回撤'] = [calculate_mdd(df_prices[c])[0] for c in df_prices.columns]
        
        numeric_cols = ['年化報酬', '年化波動', '夏普比率', '最大回撤']
        st.dataframe(res_df.style.format({c: "{:.2%}" for c in numeric_cols}), use_container_width=True)
        
        cols = st.columns(2)
        for i, col in enumerate(returns.columns):
            with cols[i%2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(returns[col], bins=40, density=True, alpha=0.7, color='steelblue')
                ax.set_title(f"{col} 日報酬分佈")
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
        st.subheader("💰 財富累積曲線 (等權重)")
        st.line_chart((1 + returns).cumprod() * initial_cap)

    with tab4:
        st.subheader("📐 市場模型 (Beta)")
        beta_data = []
        for s in [c for c in returns.columns if c not in ['0050', 'SPY']]:
            if '0050' in returns.columns and not s.isalpha(): 
                mkt_ref = '0050'
            elif 'SPY' in returns.columns:
                mkt_ref = 'SPY'
            else:
                mkt_ref = returns.columns[0] 
                
            if mkt_ref in returns.columns and s != mkt_ref:
                common_df = pd.concat([returns[mkt_ref], returns[s]], axis=1).dropna()
                if len(common_df) > 10:
                    slope, _, r_val, _, _ = stats.linregress(common_df.iloc[:,0], common_df.iloc[:,1])
                    beta_data.append({"Asset": s, "Benchmark": mkt_ref, "Beta": slope, "R2": r_val**2})
        st.table(pd.DataFrame(beta_data))

    # --- TAB 5: Scipy Optimize 效率前緣 ---#
    with tab5:
        st.subheader("⚖️ 效率前緣與最佳配置 (Scipy Optimize)")
        
        col_main, col_info = st.columns([3, 1])
        
        with col_main:
            # 1. 蒙地卡羅模擬 (背景雲)
            num_assets = len(returns.columns)
            sim_res = np.zeros((3, num_simulations))
            for i in range(num_simulations):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                p_ret, p_std = get_portfolio_performance(weights, mu, S, rf_rate)
                sim_res[0,i] = p_std
                sim_res[1,i] = p_ret
                sim_res[2,i] = (p_ret - rf_rate) / p_std 

            # 2. 數值最佳化求解
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            init_guess = num_assets * [1. / num_assets,]

            # A. 最大夏普比率組合 (Tangency Portfolio)
            opt_sharpe = sco.minimize(neg_sharpe_ratio, init_guess, args=(mu, S, rf_rate), 
                                      method='SLSQP', bounds=bounds, constraints=constraints)
            sharpe_ret, sharpe_vol = get_portfolio_performance(opt_sharpe.x, mu, S, rf_rate)
            best_weights_global = opt_sharpe.x # 更新全域變數

            # B. 最小波動率組合 (MVP)
            opt_vol = sco.minimize(minimize_volatility, init_guess, args=(mu, S, rf_rate), 
                                   method='SLSQP', bounds=bounds, constraints=constraints)
            min_vol_ret, min_vol_vol = get_portfolio_performance(opt_vol.x, mu, S, rf_rate)

            # C. 繪製效率前緣曲線 (Efficient Frontier)
            target_returns = np.linspace(min_vol_ret, max(sharpe_ret, sim_res[1].max()) * 1.05, 50)
            frontier_vol = []
            
            for t_ret in target_returns:
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                        {'type': 'eq', 'fun': lambda x: get_portfolio_performance(x, mu, S, rf_rate)[0] - t_ret})
                res = sco.minimize(minimize_volatility, init_guess, args=(mu, S, rf_rate), 
                                   method='SLSQP', bounds=bounds, constraints=cons)
                if res.success:
                    frontier_vol.append(res.fun) 
                else:
                    frontier_vol.append(np.nan)

            # 3. 繪圖
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # (1) 隨機模擬點
            sc = ax.scatter(sim_res[0,:], sim_res[1,:], c=sim_res[2,:], cmap='viridis', s=10, alpha=0.3, label='Random Portfolios')
            plt.colorbar(sc, label='Sharpe Ratio')
            
            # (2) 效率前緣線
            ax.plot(frontier_vol, target_returns, 'b-', linewidth=2.5, label='Efficient Frontier')
            
            # (3) 個別資產點
            asset_ret = mu * 252
            asset_vol = np.sqrt(np.diag(S)) * np.sqrt(252)
            ax.scatter(asset_vol, asset_ret, marker='o', color='grey', s=50, label='Assets')
            for i, txt in enumerate(returns.columns):
                ax.annotate(txt, (asset_vol[i], asset_ret[i]), xytext=(5,0), textcoords='offset points')

            # (4) 標記關鍵組合
            ax.scatter(min_vol_vol, min_vol_ret, marker='*', color='orange', s=250, edgecolors='black', label='Min Volatility (MVP)')
            ax.scatter(sharpe_vol, sharpe_ret, marker='*', color='purple', s=250, edgecolors='black', label='M
