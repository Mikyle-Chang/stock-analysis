import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from FinMind.data import DataLoader
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
    for s in list(set(tickers_tw + ['0050'])):
        try:
            ticker = f"{s}.TW"
            yf_obj = yf.Ticker(ticker)
            df = yf_obj.history(start=start, end=end, interval="1d")
            if not df.empty:
                data_dict[s] = df['Close']
        except:
            st.sidebar.warning(f"台股 {s} 抓取嘗試失敗")

    for s in tickers_us:
        try:
            yf_obj = yf.Ticker(s)
            df = yf_obj.history(start=start, end=end, interval="1d")
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
    # 這裡統一變數名稱為 num_simulations
    num_simulations = st.slider('蒙地卡羅次數', 1000, 5000, 2000)
    forecast_len = st.slider('預測天數', 30, 365, 180)

# --- 5. 主程式執行 ---
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    tw_list = [x.strip() for x in tw_in.split(',') if x.strip()]
    us_list = [x.strip().upper() for x in us_in.split(',') if x.strip()]
    
    with st.spinner('正在從 Yahoo Finance 節點抓取全球複權數據...'):
        raw_data = fetch_stock_data(tw_list, us_list, start_date, end_date)
        
        if not raw_data:
            st.error("❌ 所有來源均連線失敗。請嘗試更換日期範圍或稍後再試。")
            st.stop()
            
        df_prices = pd.DataFrame(raw_data).ffill().dropna()
        returns = df_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    st.success(f"✅ 成功載入 {len(df_prices.columns)} 檔資產數據！")
    st.download_button("📥 下載調整後數據 (CSV)", df_prices.to_csv().encode('utf-8'), "data.csv")

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
        mkt = '0050' if '0050' in returns.columns else returns.columns[0]
        beta_data = []
        for s in [c for c in returns.columns if c != mkt]:
            slope, _, r_val, _, _ = stats.linregress(returns[mkt], returns[s])
            beta_data.append({"Asset": s, "Beta": slope, "R2": r_val**2})
        st.table(pd.DataFrame(beta_data))

    with tab5:
        st.subheader("⚖️ 最佳投資組合配置")
        r_mean = returns.mean() * 252
        r_cov = returns.cov() * 252
        
        sim_res = np.zeros((3, num_simulations))
        all_weights = np.zeros((num_simulations, len(returns.columns)))
        
        for i in range(num_simulations):
            w = np.random.random(len(returns.columns))
            w /= w.sum()
            all_weights[i, :] = w
            p_r = np.sum(w * r_mean)
            p_v = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
            # 夏普比率計算修正變數名稱
            sim_res[:, i] = [p_r, p_v, (p_r - rf_rate) / p_v]
        
        tidx = np.argmax(sim_res[2])
        best_weights = all_weights[tidx, :]
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write("**效率前緣分佈圖**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10, alpha=0.5)
            ax.scatter(sim_res[1, tidx], sim_res[0, tidx], color='red', marker='*', s=200, label='最佳夏普組合')
            ax.set_xlabel("風險"); ax.set_ylabel("報酬")
            plt.colorbar(sc, label='夏普比率')
            st.pyplot(fig)

        with col2:
            st.write("**最佳資產配置比例**")
            df_weights = pd.DataFrame({'資產': returns.columns, '比例': best_weights * 100})
            df_weights = df_weights.sort_values(by='比例', ascending=False)
            
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(df_weights['比例'], labels=df_weights['資產'], autopct='%1.1f%%', startangle=140)
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
            st.dataframe(df_weights.style.format({'比例': '{:.2f}%'}))

    with tab6:
        st.subheader("🔮 股價未來模擬")
        tgt = st.selectbox("標的", returns.columns)
        s0, mu, sigma = df_prices[tgt].iloc[-1], returns[tgt].mean()*252, returns[tgt].std()*np.sqrt(252)
        dt = 1/252
        sim_paths = pd.DataFrame([s0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, forecast_len))) for _ in range(50)]).T
        st.line_chart(sim_paths)
