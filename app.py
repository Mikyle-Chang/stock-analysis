import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from FinMind.data import DataLoader
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全球投資優化系統 (調整後股價版)", layout="wide", page_icon="📈")

plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心函數 ---
def calculate_mdd(series):
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

# --- 3. 側邊欄 ---
with st.sidebar:
    st.header('1. 🎯 投資標的')
    tw_input = st.text_input('台股代號', '2330, 2454, 2317')
    us_input = st.text_input('美股代號', 'AAPL, TSLA, VOO')
    
    st.header('2. 📅 時間設定')
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    
    st.header('3. 💰 參數')
    initial_capital = st.number_input('本金', value=100000)
    rf = st.number_input('無風險利率 (%)', value=4.0) / 100.0
    num_simulations = st.slider('模擬次數', 1000, 5000, 2000)

# --- 4. 數據抓取引擎 (修正分割與調整價) ---
if st.sidebar.button('🚀 執行全方位分析', type="primary"):
    data_dict = {}
    api = DataLoader()
    
    with st.spinner('正在抓取「調整後股價」以修正分割誤差...'):
        # A. 台股：使用複權股價 (taiwan_stock_daily_adj)
        tw_stocks = [s.strip() for s in tw_input.split(',') if s.strip()]
        for s in list(set(tw_stocks + ['0050'])):
            try:
                # 這裡改用 daily_adj 以取得還原息值的價格
                df = api.taiwan_stock_daily_adj(
                    stock_id=s, 
                    start_date=start_date.strftime('%Y-%m-%d'), 
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    data_dict[s] = df.set_index('date')['close']
            except: st.warning(f"台股 {s} 抓取失敗")

        # B. 美股：使用 Adj Close (自動處理分割與股利)
        us_stocks = [s.strip().upper() for s in us_input.split(',') if s.strip()]
        if us_stocks:
            try:
                # yfinance 的 download 預設會包含 Adj Close
                us_data = yf.download(us_stocks, start=start_date, end=end_date)
                # 確保取用 'Adj Close' 欄位
                if 'Adj Close' in us_data.columns:
                    adj_close = us_data['Adj Close']
                    if isinstance(adj_close, pd.Series):
                        data_dict[us_stocks[0]] = adj_close
                    else:
                        for c in adj_close.columns:
                            data_dict[c] = adj_close[c]
            except: st.error("美股抓取失敗")

        if not data_dict:
            st.error("❌ 無效數據")
            st.stop()
            
        # 合併與清洗
        df_all_prices = pd.DataFrame(data_dict).ffill().dropna()
        returns = df_all_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
    # --- 功能區：CSV 下載 ---
    st.success(f"✅ 已完成分割調整！共 {len(df_all_prices)} 筆數據")
    st.download_button("📥 下載調整後價格數據 (CSV)", df_all_prices.to_csv().encode('utf-8'), "adj_data.csv")

    # --- 分頁內容 (保持完整功能) ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 統計", "🔗 相關性", "💰 累積財富", "📐 Beta/CAPM", "⚖️ 效率前緣", "🔮 預測"
    ])

    with tab1:
        # 統計表格與直方圖 (邏輯同前，確保數據 clean)
        stats_df = pd.DataFrame(index=returns.columns)
        stats_df['年化報酬'] = returns.mean() * 252
        stats_df['最大回撤'] = [calculate_mdd(df_all_prices[c])[0] for c in df_all_prices.columns]
        st.dataframe(stats_df.style.format("{:.2%}"), use_container_width=True)
        
        cols = st.columns(2)
        for i, asset in enumerate(returns.columns):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(returns[asset], bins=40, density=True, alpha=0.7)
                st.pyplot(fig)

    with tab2:
        # 相關性矩陣 (放大版)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(returns.corr(), cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        ax.set_xticks(range(len(returns.columns))); ax.set_xticklabels(returns.columns, rotation=45)
        ax.set_yticks(range(len(returns.columns))); ax.set_yticklabels(returns.columns)
        st.pyplot(fig)

    with tab3:
        st.line_chart((1 + returns).cumprod() * initial_capital)

    with tab4:
        # 市場模型 (Beta)
        mkt = '0050' if '0050' in returns.columns else returns.columns[0]
        beta_res = []
        for s in [c for c in returns.columns if c != mkt]:
            slope, _, r_val, _, _ = stats.linregress(returns[mkt], returns[s])
            beta_res.append({"標的": s, "Beta": slope, "R2": r_val**2})
        st.table(pd.DataFrame(beta_res))

    with tab5:
        # 效率前緣
        r_mean, r_cov = returns.mean()*252, returns.cov()*252
        results = np.zeros((3, num_simulations))
        for i in range(num_simulations):
            w = np.random.random(len(returns.columns)); w /= w.sum()
            p_r = np.sum(w * r_mean); p_v = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
            results[:, i] = [p_r, p_v, (p_r - rf) / p_v]
        st.pyplot(plt.subplots()[1].scatter(results[1], results[0], c=results[2], s=10).figure)

    with tab6:
        # 預測
        tgt = st.selectbox("預測標的", returns.columns)
        s0, mu, sigma = df_all_prices[tgt].iloc[-1], returns[tgt].mean()*252, returns[tgt].std()*np.sqrt(252)
        paths = pd.DataFrame([s0 * np.exp(np.cumsum((mu-0.5*sigma**2)*(1/252) + sigma*np.sqrt(1/252)*np.random.normal(0,1,forecast_days))) for _ in range(50)]).T
        st.line_chart(paths)
