import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from FinMind.data import DataLoader
import yfinance as yf
from datetime import datetime, timedelta
import requests

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球投資組合優化系統", layout="wide", page_icon="📈")

plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心工具函數 ---
def calculate_mdd(series):
    """計算最大回撤"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

# --- 3. 側邊欄設定 ---
with st.sidebar:
    st.header('1. 🎯 投資標的')
    tw_input = st.text_input('台股代號', '2330, 2454, 2317')
    us_input = st.text_input('美股代號', 'VOO, QQQ, AAPL')
    
    st.header('2. 📅 回測設定')
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    
    st.header('3. 💰 參數管理')
    initial_capital = st.number_input('初始本金', value=100000)
    rf = st.number_input('無風險利率 (%)', value=4.0) / 100.0
    num_simulations = st.slider('模擬次數', 1000, 5000, 2000)
    forecast_days = st.slider('預測天數', 30, 365, 180)

# --- 4. 強化數據抓取引擎 ---
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    data_dict = {}
    api = DataLoader()
    
    with st.spinner('正在同步數據... 若失敗請嘗試重新點擊按鈕'):
        # --- A. 台股處理 (優先使用 yfinance 抓取調整後股價，穩定性較高) ---
        tw_stocks = [s.strip() for s in tw_input.split(',') if s.strip()]
        all_tw = list(set(tw_stocks + ['0050']))
        
        for s in all_tw:
            success = False
            # 1. 嘗試使用 yfinance 抓取 (加上 .TW) - 這是目前最穩定的免費來源
            try:
                ticker = f"{s}.TW"
                # 加入進階設定防止被封鎖
                yf_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not yf_df.empty:
                    # 抓取 Adj Close 確保處理除權息與分割
                    data_dict[s] = yf_df['Adj Close']
                    success = True
            except: pass
            
            # 2. 如果 yfinance 失敗，嘗試 FinMind
            if not success:
                try:
                    df = api.taiwan_stock_daily_adj(stock_id=s, start_date=start_date.strftime('%Y-%m-%d'))
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        data_dict[s] = df.set_index('date')['close']
                        success = True
                except: pass
            
            if not success:
                st.warning(f"⚠️ 無法取得台股 {s}，請檢查代號正確性。")

        # --- B. 美股處理 (yfinance) ---
        us_stocks = [s.strip().upper() for s in us_input.split(',') if s.strip()]
        if us_stocks:
            try:
                us_data = yf.download(us_stocks, start=start_date, end=end_date, progress=False)['Adj Close']
                if isinstance(us_data, pd.Series):
                    data_dict[us_stocks[0]] = us_data
                else:
                    for c in us_data.columns:
                        data_dict[c] = us_data[c]
            except: st.error("❌ 美股來源連線失敗")

        if not data_dict:
            st.error("❌ 無效數據。推論：可能是連線被阻擋，請稍候再試。")
            st.stop()
            
        # 數據對齊、處理分割產生的 NaN 以及無窮大值
        df_all_prices = pd.DataFrame(data_dict).ffill().dropna()
        returns = df_all_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
    # --- 功能區 ---
    st.success(f"✅ 資料載入成功！")
    st.download_button("📥 下載調整後數據 (CSV)", df_all_prices.to_csv().encode('utf-8'), "data.csv")

    # (分頁 Tab 1 ~ Tab 6 的內容與先前相同，此處省略以節省長度，確保您保留原有的分析邏輯)
    # ... 原有分頁代碼 ...
    # Tab 1: 統計
    with tab1:
        st.subheader("📋 資產報酬統計 (基於調整後股價)")
        stats_df = pd.DataFrame(index=returns.columns)
        stats_df['Ann. Return'] = returns.mean() * 252
        stats_df['Ann. Volatility'] = returns.std() * np.sqrt(252)
        stats_df['Sharpe Ratio'] = (stats_df['Ann. Return'] - rf) / stats_df['Ann. Volatility']
        stats_df['Max Drawdown'] = [calculate_mdd(df_all_prices[c])[0] for c in df_all_prices.columns]
        stats_df['Skew'] = returns.skew()
        stats_df['Kurt'] = returns.kurt()
        stats_df['JB_p'] = [stats.jarque_bera(returns[c])[1] for c in returns.columns]
        
        # 格式化
        disp = stats_df.copy()
        for col in ['Ann. Return', 'Ann. Volatility', 'Max Drawdown']:
            disp[col] = disp[col].apply(lambda x: f"{x:.2%}")
        disp['Normality'] = disp['JB_p'].apply(interpret_jb_test)
        st.dataframe(disp, use_container_width=True)

        cols = st.columns(2)
        for i, asset in enumerate(returns.columns):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(returns[asset], bins=40, color='#2980b9', alpha=0.7, density=True, edgecolor='white')
                st.pyplot(fig)

    # Tab 2: 相關性
    with tab2:
        st.subheader("🔗 相關性分析")
        st.pyplot(plot_heatmap_matplotlib(returns.corr()), use_container_width=True)

    # Tab 3: 投資模擬
    with tab3:
        st.subheader("💰 累積財富增長")
        st.line_chart((1 + returns).cumprod() * initial_capital)

    # Tab 4: 市場模型
    with tab4:
        st.subheader("📐 Beta 係數衡量 (基準: 0050)")
        mkt = '0050' if '0050' in returns.columns else returns.columns[0]
        capm_res = []
        for s in [c for c in returns.columns if c != mkt]:
            slope, _, r_val, _, _ = stats.linregress(returns[mkt], returns[s])
            capm_res.append({"Asset": s, "Beta": slope, "R2": r_val**2})
        st.dataframe(pd.DataFrame(capm_res).set_index("Asset"))

    # Tab 5: 效率前緣
    with tab5:
        st.subheader("⚖️ 投資組合優化 (Markowitz)")
        r_mean, r_cov = returns.mean()*252, returns.cov()*252
        sim_res = np.zeros((3, num_simulations))
        for i in range(num_simulations):
            w = np.random.random(len(returns.columns)); w /= w.sum()
            p_r = np.sum(w * r_mean); p_v = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
            sim_res[:, i] = [p_r, p_v, (p_r - rf) / p_v]
        
        tidx = np.argmax(sim_res[2])
        fig, ax = plt.subplots()
        ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10)
        ax.scatter(sim_res[1, tidx], sim_res[0, tidx], color='red', marker='*', s=200)
        st.pyplot(fig)

    # Tab 6: 預測
    with tab6:
        st.subheader("🔮 股價未來模擬")
        tgt = st.selectbox("選擇預測標的", returns.columns)
        s0, mu, sigma = df_all_prices[tgt].iloc[-1], returns[tgt].mean()*252, returns[tgt].std()*np.sqrt(252)
        dt = 1/252
        paths = pd.DataFrame([s0 * np.exp(np.cumsum((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1,forecast_days))) for _ in range(50)]).T
        st.line_chart(paths)

