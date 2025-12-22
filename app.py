import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球投資組合分析系統", layout="wide", page_icon="📈")

# 設定中文字體 (嘗試兼容不同系統)
plt.style.use('bmh')
font_names = [f.name for f in fm.fontManager.ttflist]
if 'Microsoft JhengHei' in font_names:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif 'Arial Unicode MS' in font_names:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
elif 'SimHei' in font_names:
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['sans-serif'] # Fallback
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心計算函數 ---
def calculate_mdd(series):
    """計算最大回撤"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

# --- 3. 強化型數據抓取函數 (修正點 1: auto_adjust=True) ---
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers_tw, tickers_us, start, end):
    data_dict = {}
    
    # 建立抓取清單，加入 0050 作為潛在的 Beta 基準
    unique_tw = list(set(tickers_tw + ['0050']))
    
    # 下載台股
    for s in unique_tw:
        if not s: continue
        try:
            ticker = f"{s}.TW"
            yf_obj = yf.Ticker(ticker)
            # 關鍵修正：使用 auto_adjust=True 獲取還原權值股價
            df = yf_obj.history(start=start, end=end, interval="1d", auto_adjust=True)
            if not df.empty:
                data_dict[s] = df['Close']
        except:
            st.sidebar.warning(f"台股 {s} 抓取失敗")

    # 下載美股
    for s in tickers_us:
        if not s: continue
        try:
            yf_obj = yf.Ticker(s)
            # 關鍵修正：使用 auto_adjust=True
            df = yf_obj.history(start=start, end=end, interval="1d", auto_adjust=True)
            if not df.empty:
                data_dict[s] = df['Close']
        except:
            st.sidebar.warning(f"美股 {s} 抓取失敗")
            
    return data_dict

# --- 4. 側邊欄 ---
with st.sidebar:
    st.header('🎯 標的設定')
    tw_in = st.text_input('台股代號', '1215,2330,2412,2886')
    us_in = st.text_input('美股代號', 'SPY,QQQ,TLT,GLD')
    
    st.header('📅 時間與資金')
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    initial_cap = st.number_input('本金', value=100000)
    rf_rate = st.number_input('無風險利率 (%)', value=2.0) / 100
    
    st.header('🎲 模擬設定')
    num_simulations = st.slider('蒙地卡羅次數', 1000, 5000, 2000)
    forecast_len = st.slider('預測天數', 30, 365, 180)

# --- 5. 主程式執行 ---
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    tw_list = [x.strip() for x in tw_in.split(',') if x.strip()]
    us_list = [x.strip().upper() for x in us_in.split(',') if x.strip()]
    
    with st.spinner('正在抓取還原權值數據 (Total Return)...'):
        # 修正點 3: 移除了 FinMind 相關引用，僅使用 yfinance
        raw_data = fetch_stock_data(tw_list, us_list, start_date, end_date)
        
        if not raw_data:
            st.error("❌ 無法獲取任何數據。請檢查代號或網路連線。")
            st.stop()
            
        df_prices = pd.DataFrame(raw_data).ffill().dropna()
        if df_prices.shape[0] < 30:
            st.error("❌ 有效交易日過少，無法進行分析。")
            st.stop()
            
        returns = df_prices.pct_change().dropna()

    st.success(f"✅ 成功載入 {len(df_prices.columns)} 檔資產數據！ (已還原權息)")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 統計", "🔗 相關性", "💰 模擬", "📐 Beta分析", "⚖️ 效率前緣", "🔮 預測"])

    with tab1:
        st.subheader("📋 統計特徵 (年化)")
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
        st.subheader("💰 財富累積曲線")
        st.line_chart((1 + returns).cumprod() * initial_cap)

    with tab4:
        st.subheader("📐 市場模型 (Beta)")
        
        # 修正點 4: 智慧基準選擇
        if 'SPY' in returns.columns:
            mkt_benchmark = 'SPY'
        elif '0050' in returns.columns:
            mkt_benchmark = '0050'
        else:
            mkt_benchmark = returns.columns[0]
            
        st.info(f"目前使用的市場基準 (Benchmark): **{mkt_benchmark}**")
        
        beta_data = []
        for s in [c for c in returns.columns if c != mkt_benchmark]:
            # 確保對齊後再計算
            common = pd.concat([returns[mkt_benchmark], returns[s]], axis=1).dropna()
            if len(common) > 10:
                slope, _, r_val, _, _ = stats.linregress(common.iloc[:,0], common.iloc[:,1])
                beta_data.append({"Asset": s, "Beta": slope, "R2": r_val**2})
        
        if beta_data:
            st.table(pd.DataFrame(beta_data).set_index("Asset"))
        else:
            st.write("無足夠資產進行 Beta 計算。")

    with tab5:
        st.subheader("⚖️ 效率前緣 (Efficient Frontier)")
        # 使用矩陣運算加速
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        weights = np.random.random((num_simulations, len(returns.columns)))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        
        port_returns = np.dot(weights, mean_returns)
        port_vols = np.sqrt(np.einsum('ij,ji->i', np.dot(weights, cov_matrix), weights.T))
        port_sharpe = (port_returns - rf_rate) / port_vols
        
        max_idx = np.argmax(port_sharpe)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(port_vols, port_returns, c=port_sharpe, cmap='viridis', s=10, alpha=0.5)
            ax.scatter(port_vols[max_idx], port_returns[max_idx], c='red', marker='*', s=200, label='Max Sharpe')
            ax.set_xlabel("Risk (Vol)"); ax.set_ylabel("Return")
            plt.colorbar(sc)
            st.pyplot(fig)

        with col2:
            st.write("**最佳配置**")
            best_w = weights[max_idx]
            df_best = pd.DataFrame({'Asset': returns.columns, 'Weight': best_w})
            df_best = df_best.sort_values(by='Weight', ascending=False)
            
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(df_best['Weight'], labels=df_best['Asset'], autopct='%1.1f%%')
            st.pyplot(fig_pie)
            st.dataframe(df_best.style.format({'Weight': '{:.2%}'}))

    with tab6:
        st.subheader("🔮 股價幾何布朗運動 (GBM) 模擬")
        tgt = st.selectbox("標的", returns.columns)
        
        # 修正點 5: 正確的 GBM 漂移項與向量化模擬
        s0 = df_prices[tgt].iloc[-1]
        mu = returns[tgt].mean() * 252
        sigma = returns[tgt].std() * np.sqrt(252)
        dt = 1/252
        
        # 建立模擬路徑矩陣 (TimeSteps x Simulations)
        paths = np.zeros((forecast_len, 50))
        paths[0] = s0
        
        # 修正後的漂移項公式
        drift = (mu - 0.5 * sigma**2) * dt
        shock_scale = sigma * np.sqrt(dt)
        
        for t in range(1, forecast_len):
            z = np.random.normal(0, 1, 50)
            paths[t] = paths[t-1] * np.exp(drift + shock_scale * z)
            
        st.line_chart(paths)
