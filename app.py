import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import os

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球投資組合分析系統", layout="wide", page_icon="📈")

# --- 🎯 雲端通用字體解決方案 ---
def set_font():
    # 下載或指定專案資料夾內的字體檔
    font_path = 'NotoSansTC-Regular.ttf' 
    if os.path.exists(font_path):
        # 載入字體並設定為 Matplotlib 預設
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        # 加入這行確保 Matplotlib 註冊了該字體
        fm.fontManager.addfont(font_path)
    else:
        # 如果沒檔案，嘗試最後的掙扎（針對 Linux 環境）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        st.sidebar.warning("找不到 NotoSansTC-Regular.ttf，中文字體可能無法顯示")
    
    plt.rcParams['axes.unicode_minus'] = False 

set_font()
plt.style.use('bmh')

# --- 剩下程式碼保持完全不動 (2. 核心計算函數以後...) ---
# --- 2. 核心計算函數 ---
def calculate_mdd(series):
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

# --- 3. 數據抓取函數 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers_tw, tickers_us, start, end):
    data_dict = {}
    for s in list(set(tickers_tw + ['0050'])):
        try:
            df = yf.Ticker(f"{s}.TW").history(start=start, end=end, auto_adjust=True)
            if not df.empty: data_dict[s] = df['Close']
        except: st.sidebar.warning(f"台股 {s} 失敗")
    for s in list(set(tickers_us + ['SPY'])):
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
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    initial_cap = st.number_input('本金', value=100000)
    rf_rate = st.number_input('無風險利率 (%)', value=4.0) / 100
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
        res_df['年化報酬'] = returns.mean() * 252
        res_df['年化波動'] = returns.std() * np.sqrt(252)
        res_df['夏普比率'] = (res_df['年化報酬'] - rf_rate) / res_df['年化波動']
        res_df['最大回撤'] = [calculate_mdd(df_prices[c])[0] for c in df_prices.columns]
        st.dataframe(res_df.style.format("{:.2%}"), use_container_width=True)
        cols = st.columns(2)
        for i, col in enumerate(returns.columns):
            with cols[i%2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(returns[col], bins=40, density=True, alpha=0.7, color='steelblue')
                ax.set_title(f"{col} 報酬率分佈")
                st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(returns.corr(), cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im); st.pyplot(fig)

    with tab3:
        st.line_chart((1 + returns).cumprod() * initial_cap)

    with tab4:
        st.subheader("📐 市場模型 (Beta)")
        beta_data = []
        for s in returns.columns:
            mkt = 'SPY' if s in us_list else '0050'
            if s == mkt: continue
            common = pd.concat([returns[mkt], returns[s]], axis=1).dropna()
            if len(common) > 10:
                slope, _, r_val, _, _ = stats.linregress(common.iloc[:,0], common.iloc[:,1])
                beta_data.append({"Asset": s, "Benchmark": mkt, "Beta": slope, "R2": r_val**2})
        st.table(pd.DataFrame(beta_data))

    tw_assets = [s for s in returns.columns if s in tw_list or s == '0050']
    best_weights_final = None
    
    with tab5:
        st.subheader("⚖️ 最佳投資組合配置 (僅台股)")
        if len(tw_assets) >= 2:
            tw_returns = returns[tw_assets]
            r_mean, r_cov = tw_returns.mean() * 252, tw_returns.cov() * 252
            sim_res = np.zeros((3, num_simulations))
            all_weights = np.zeros((num_simulations, len(tw_assets)))
            for i in range(num_simulations):
                w = np.random.random(len(tw_assets))
                w /= w.sum(); all_weights[i, :] = w
                p_r = np.sum(w * r_mean)
                p_v = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
                sim_res[:, i] = [p_r, p_v, (p_r - rf_rate) / p_v]
            tidx = np.argmax(sim_res[2])
            best_weights_final = all_weights[tidx, :]
            col1, col2 = st.columns([3, 2])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sc = ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10, alpha=0.5)
                ax.scatter(sim_res[1, tidx], sim_res[0, tidx], color='red', marker='*', s=200, label='最佳夏普點')
                cml_x = np.linspace(0, max(sim_res[1])*1.2, 100)
                ax.plot(cml_x, rf_rate + sim_res[2, tidx] * cml_x, color='darkorange', linestyle='--', label='資本市場線')
                ax.set_title("效率前緣分析 (台股組合)"); ax.legend(); st.pyplot(fig)
            with col2:
                df_w = pd.DataFrame({'資產': tw_assets, '比例': best_weights_final * 100})
                st.dataframe(df_w.sort_values(by='比例', ascending=False).style.format({'比例': '{:.2f}%'}))
        else: st.warning("台股數量不足。")

    with tab6:
        st.subheader("🔮 最佳組合未來財富模擬")
        if best_weights_final is not None:
            port_returns = (returns[tw_assets] * best_weights_final).sum(axis=1)
            mu, sigma = port_returns.mean() * 252, port_returns.std() * np.sqrt(252)
            s0, dt = initial_cap, 1/252
            sim_paths = np.zeros((forecast_len, 50))
            sim_paths[0] = s0
            for t in range(1, forecast_len):
                sim_paths[t] = sim_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, 50))
            st.write(f"預測年化報酬: {mu:.2%}, 年化波動: {sigma:.2%}")
            st.line_chart(sim_paths)

