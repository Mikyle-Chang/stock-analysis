import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from FinMind.data import DataLoader
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 頁面與視覺設定 ---
st.set_page_config(page_title="全球投資組合分析系統", layout="wide", page_icon="📈")

# 設定圖表風格與中文字體
plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# CSS 美化
st.markdown("""
    <style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🎓 投資組合分析系統 (Final Project)')
st.caption("已整合：台美股真實數據、MDD、CSV下載、完整統計模組")

# --- 2. 核心工具函數 ---
def interpret_jb_test(p_value):
    return "❌ 拒絕常態" if p_value < 0.05 else "✅ 近似常態"

def calculate_mdd(series):
    """計算最大回撤"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

def plot_heatmap_matplotlib(df_corr):
    """超大尺寸相關性熱力圖"""
    fig, ax = plt.subplots(figsize=(14, 12)) 
    cax = ax.imshow(df_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)
    ticks = np.arange(len(df_corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df_corr.columns, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(df_corr.index, fontsize=12)
    for i in range(len(df_corr.columns)):
        for j in range(len(df_corr.columns)):
            val = df_corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                    color="white" if abs(val) > 0.5 else "black", fontweight='bold')
    ax.set_title("資產相關係數矩陣", fontsize=18)
    return fig

# --- 3. 側邊欄設定 ---
with st.sidebar:
    st.header('1. 🎯 投資標的')
    tw_input = st.text_input('台股代號 (如: 2330, 2454)', '2330, 2454, 2317')
    us_input = st.text_input('美股代號 (如: VOO, QQQ, AAPL)', 'VOO, QQQ')
    
    st.header('2. 📅 回測設定')
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    
    st.header('3. 💰 資金管理')
    initial_capital = st.number_input('初始投入本金', value=100000)
    risk_free_rate_pct = st.number_input('無風險利率 (%)', value=4.0)
    rf = risk_free_rate_pct / 100.0
    
    st.header('4. 🎲 模型參數')
    num_simulations = st.slider('蒙地卡羅模擬次數', 1000, 10000, 3000)
    forecast_days = st.slider('未來預測天數', 30, 365, 180)

# --- 4. 數據抓取主程式 ---
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    data_dict = {}
    
    with st.spinner('正在同步全球市場真實數據...'):
        # A. 台股抓取 (FinMind)
        api = DataLoader()
        tw_stocks = [s.strip() for s in tw_input.split(',') if s.strip()]
        # 確保抓取 0050 作為市場基準
        for s in list(set(tw_stocks + ['0050'])):
            try:
                df = api.taiwan_stock_daily(stock_id=s, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    data_dict[s] = df.set_index('date')['close']
            except: pass

        # B. 美股抓取 (yfinance)
        us_stocks = [s.strip().upper() for s in us_input.split(',') if s.strip()]
        if us_stocks:
            try:
                us_data = yf.download(us_stocks, start=start_date, end=end_date)['Close']
                if isinstance(us_data, pd.Series):
                    data_dict[us_stocks[0]] = us_data
                else:
                    for c in us_data.columns:
                        data_dict[c] = us_data[c]
            except: st.error("美股抓取出現異常")

        if not data_dict:
            st.error("❌ 無法抓取任何資料，請檢查代號或網路。")
            st.stop()
            
        # 數據清理與對齊
        df_all_prices = pd.DataFrame(data_dict).ffill().dropna()
        # 解決 ValueError: 強制過濾非有限數值
        returns = df_all_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        st.success(f"✅ 資料分析完成！期間：{start_date} ~ {end_date} (共 {len(df_all_prices)} 交易日)")

    # --- 下載區 ---
    csv_data = df_all_prices.to_csv().encode('utf-8')
    st.download_button(label="📥 下載原始價格數據 (CSV)", data=csv_data, file_name='market_data.csv', mime='text/csv')

    # --- 分頁標籤 (保留所有原始功能) ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 A. 統計特徵", "🔗 B. 相關性矩陣", "💰 C. 投資模擬", 
        "📐 D. 市場模型", "⚖️ E. 效率前緣", "🔮 F. 未來預測"
    ])

    # --- Tab 1: 統計特徵 ---
    with tab1:
        st.subheader("📊 資產報酬統計")
        stats_df = pd.DataFrame(index=returns.columns)
        stats_df['Ann. Return'] = returns.mean() * 252
        stats_df['Total Return'] = (df_all_prices.iloc[-1] / df_all_prices.iloc[0]) - 1
        stats_df['Ann. Volatility'] = returns.std() * np.sqrt(252)
        stats_df['Skew'] = returns.skew()
        stats_df['Kurt'] = returns.kurt()
        stats_df['JB_p'] = [stats.jarque_bera(returns[c])[1] for c in returns.columns]
        stats_df['Max Drawdown'] = [calculate_mdd(df_all_prices[c])[0] for c in df_all_prices.columns]

        display_df = stats_df.copy()
        for col in ['Ann. Return', 'Total Return', 'Ann. Volatility', 'Max Drawdown']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        display_df['Normality'] = display_df['JB_p'].apply(interpret_jb_test)
        st.dataframe(display_df, use_container_width=True)

        st.divider()
        st.subheader("📉 分布直方圖")
        cols = st.columns(2)
        for i, asset in enumerate(returns.columns):
            asset_data = returns[asset]
            if np.isfinite(asset_data).all(): # 再次防呆
                with cols[i % 2]:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(asset_data, bins=40, color='#2980b9', alpha=0.7, density=True, edgecolor='white')
                    # 加入常態曲線
                    mu, std = asset_data.mean(), asset_data.std()
                    x = np.linspace(asset_data.min(), asset_data.max(), 100)
                    ax.plot(x, stats.norm.pdf(x, mu, std), 'r', linewidth=2)
                    ax.set_title(f"{asset} 報酬率分佈")
                    st.pyplot(fig)

    # --- Tab 2: 相關性 ---
    with tab2:
        st.subheader("🔗 12x12 (或更多) 相關性矩陣")
        st.pyplot(plot_heatmap_matplotlib(returns.corr()), use_container_width=True)

    # --- Tab 3: 投資模擬 ---
    with tab3:
        st.subheader("💰 累積財富曲線")
        cum_wealth = (1 + returns).cumprod() * initial_capital
        st.line_chart(cum_wealth)
        st.write("**期末價值排名:**")
        st.dataframe(cum_wealth.iloc[-1].sort_values(ascending=False).to_frame(name="Final Value").style.format("${:,.0f}"))

    # --- Tab 4: 市場模型 ---
    with tab4:
        st.subheader("📐 市場模型風險衡量 (CAPM)")
        # 尋找基準點，優先用 0050，否則用第一個
        mkt = '0050' if '0050' in returns.columns else returns.columns[0]
        mkt_ret = returns[mkt]
        res = []
        for stock in [s for s in returns.columns if s != mkt]:
            slope, intercept, r_val, p_val, _ = stats.linregress(mkt_ret, returns[stock])
            res.append({"Asset": stock, "Beta": slope, "Alpha": intercept, "R-Squared": r_val**2})
        st.dataframe(pd.DataFrame(res).set_index("Asset").style.background_gradient(cmap='Oranges'))

    # --- Tab 5: 效率前緣 ---
    with tab5:
        st.subheader("⚖️ 效率前緣 (Monte Carlo)")
        # 排除 0050 做組合優化
        risky = [c for c in returns.columns if c != '0050']
        if len(risky) < 2:
            st.warning("請至少輸入兩個標的（除 0050 外）來進行組合優化。")
        else:
            r_mean = returns[risky].mean() * 252
            r_cov = returns[risky].cov() * 252
            sim_res = np.zeros((3, num_simulations))
            for i in range(num_simulations):
                w = np.random.random(len(risky)); w /= w.sum()
                p_ret = np.sum(w * r_mean)
                p_std = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
                sim_res[:, i] = [p_ret, p_std, (p_ret - rf) / p_std]
            
            tidx = np.argmax(sim_res[2])
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10, alpha=0.5)
            ax.scatter(sim_res[1, tidx], sim_res[0, tidx], c='red', marker='*', s=200, label='Best Sharpe')
            ax.set_xlabel("風險 (Volatility)"); ax.set_ylabel("預期報酬")
            plt.colorbar(sc, label='Sharpe Ratio')
            st.pyplot(fig)

    # --- Tab 6: 未來預測 ---
    with tab6:
        st.subheader("🔮 股價幾何布朗運動模擬")
        tgt = st.selectbox("選擇預測標的", returns.columns)
        s0 = df_all_prices[tgt].iloc[-1]
        mu = returns[tgt].mean() * 252
        sigma = returns[tgt].std() * np.sqrt(252)
        
        dt = 1/252
        sim_df = pd.DataFrame()
        for x in range(100): # 100 條路徑
            path = s0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, forecast_days)))
            sim_df[f'Path_{x}'] = path
        
        st.line_chart(sim_df)
        st.write(f"預測 {forecast_days} 天後平均價格： {np.mean(sim_df.iloc[-1]):.2f}")
