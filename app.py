import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球投資組合分析系統", layout="wide", page_icon="📈")

plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心計算函數 ---
def calculate_mdd(series):
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

# --- 3. 數據抓取函數 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers_tw, tickers_us, start, end):
    data_dict = {}
    # 強制抓取兩個基準點
    unique_tw = list(set(tickers_tw + ['0050']))
    unique_us = list(set(tickers_us + ['SPY']))
    
    for s in unique_tw:
        try:
            df = yf.Ticker(f"{s}.TW").history(start=start, end=end, auto_adjust=True)
            if not df.empty: data_dict[s] = df['Close']
        except: st.sidebar.warning(f"台股 {s} 失敗")

    for s in unique_us:
        try:
            df = yf.Ticker(s).history(start=start, end=end, auto_adjust=True)
            if not df.empty: data_dict[s] = df['Close']
        except: st.sidebar.warning(f"美股 {s} 失敗")
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
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    tw_list = [x.strip() for x in tw_in.split(',') if x.strip()]
    us_list = [x.strip().upper() for x in us_in.split(',') if x.strip()]
    
    with st.spinner('抓取數據中...'):
        raw_data = fetch_stock_data(tw_list, us_list, start_date, end_date)
        if not raw_data: st.stop()
        df_prices = pd.DataFrame(raw_data).ffill().dropna()
        returns = df_prices.pct_change().dropna()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 統計", "🔗 相關性", "💰 模擬", "📐 市場模型", "⚖️ 效率前緣", "🔮 預測"])

    with tab1:
        st.subheader("📋 統計特徵")
        res_df = pd.DataFrame(index=returns.columns)
        # 修正：改用幾何平均 (CAGR) 以符合事實
        years = (df_prices.index[-1] - df_prices.index[0]).days / 365.25
        res_df['年化報酬'] = (df_prices.iloc[-1] / df_prices.iloc[0])**(1/years) - 1
        res_df['年化波動'] = returns.std() * np.sqrt(252)
        res_df['夏普比率'] = (res_df['年化報酬'] - rf_rate) / res_df['年化波動']
        res_df['最大回撤'] = [calculate_mdd(df_prices[c])[0] for c in df_prices.columns]
        st.dataframe(res_df.style.format("{:.2%}"), use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(returns.corr(), cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im); st.pyplot(fig)

    with tab3:
        st.line_chart((1 + returns).cumprod() * initial_cap)

    with tab4:
        st.subheader("📐 市場模型 (分區 Beta)")
        beta_data = []
        for s in returns.columns:
            # 判斷基準：美股代號(不含點)用 SPY，其餘(台股)用 0050
            mkt_ref = 'SPY' if s in us_list else '0050'
            if s == mkt_ref: continue
            common = pd.concat([returns[mkt_ref], returns[s]], axis=1).dropna()
            slope, _, r_val, _, _ = stats.linregress(common.iloc[:,0], common.iloc[:,1])
            beta_data.append({"資產": s, "基準": mkt_ref, "Beta": slope, "R2": r_val**2})
        st.table(pd.DataFrame(beta_data))

    with tab5:
        st.subheader("⚖️ 效率前緣與夏普分析")
        r_mean = returns.mean() * 252
        r_cov = returns.cov() * 252
        
        sim_res = np.zeros((3, num_simulations))
        all_weights = np.zeros((num_simulations, len(returns.columns)))
        
        for i in range(num_simulations):
            w = np.random.random(len(returns.columns))
            w /= w.sum(); all_weights[i, :] = w
            p_r = np.sum(w * r_mean)
            p_v = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
            sim_res[:, i] = [p_r, p_v, (p_r - rf_rate) / p_v]
        
        tidx = np.argmax(sim_res[2]) # 最大夏普
        mvp_idx = np.argmin(sim_res[1]) # 最小變異
        
        st.metric("最佳夏普值 (Max Sharpe Ratio)", f"{sim_res[2, tidx]:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10, alpha=0.3)
        # 畫點
        ax.scatter(sim_res[1, tidx], sim_res[0, tidx], color='red', marker='*', s=200, label='最佳夏普組合')
        ax.scatter(sim_res[1, mvp_idx], sim_res[0, mvp_idx], color='blue', marker='X', s=150, label='最小變異組合')
        # 畫線 (資本市場線 CML)
        cml_x = [0, sim_res[1, tidx] * 1.5]
        cml_y = [rf_rate, rf_rate + sim_res[2, tidx] * cml_x[1]]
        ax.plot(cml_x, cml_y, color='darkorange', linestyle='--', linewidth=2, label='資本市場線')
        
        ax.set_xlabel("年化波動度 (風險)"); ax.set_ylabel("預期報酬率")
        ax.set_xlim(left=0); ax.legend(); st.pyplot(fig)

    with tab6:
        tgt = st.selectbox("標的", returns.columns)
        s0, mu, sigma = df_prices[tgt].iloc[-1], returns[tgt].mean()*252, returns[tgt].std()*np.sqrt(252)
        dt = 1/252
        sim_paths = np.zeros((forecast_len, 50))
        sim_paths[0] = s0
        for t in range(1, forecast_len):
            sim_paths[t] = sim_paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1,50))
        st.line_chart(sim_paths)
