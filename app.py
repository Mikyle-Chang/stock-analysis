import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from FinMind.data import DataLoader
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. 頁面與視覺設定 (保留原樣) ---
st.set_page_config(page_title="投資組合系統", layout="wide", page_icon="📈")

plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

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

st.title('🎓 投資組合分析系統 (Final Project - 全球實測版)')

# --- 2. 側邊欄 (還原並擴充) ---
with st.sidebar:
    st.header('1. 🎯 投資標的')
    tw_input = st.text_input('台股代號 (如: 2330, 2454)', '2330, 2454, 2317')
    us_input = st.text_input('美股代號 (如: VOO, QQQ, TSLA)', 'VOO, QQQ')
    
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

# --- 3. 核心函數 (保留原有邏輯) ---
def interpret_jb_test(p_value):
    return "❌ 拒絕常態" if p_value < 0.05 else "✅ 近似常態"

def calculate_mdd(series):
    """新增：最大回撤計算"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

def plot_heatmap_matplotlib(df_corr):
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(df_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)
    ticks = np.arange(len(df_corr.columns))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(df_corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(df_corr.index)
    for i in range(len(df_corr.columns)):
        for j in range(len(df_corr.columns)):
            ax.text(j, i, f"{df_corr.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(df_corr.iloc[i,j]) > 0.5 else "black")
    return fig

# --- 4. 主程式 ---
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    data_dict = {}
    
    with st.spinner('正在同步全球市場數據...'):
        # 抓台股 (FinMind)
        api = DataLoader()
        tw_stocks = [s.strip() for s in tw_input.split(',') if s.strip()]
        for s in tw_stocks + ['0050']:
            try:
                df = api.taiwan_stock_daily(stock_id=s, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    data_dict[f"{s}"] = df.set_index('date')['close']
            except: pass

        # 抓美股 (yfinance)
        us_stocks = [s.strip().upper() for s in us_input.split(',') if s.strip()]
        if us_stocks:
            try:
                us_data = yf.download(us_stocks, start=start_date, end=end_date)['Close']
                if isinstance(us_data, pd.Series):
                    data_dict[us_stocks[0]] = us_data
                else:
                    for c in us_data.columns: data_dict[c] = us_data[c]
            except: st.error("美股抓取失敗")

        if not data_dict:
            st.error("無法抓取資料。")
            st.stop()
            
        df_all_prices = pd.DataFrame(data_dict).ffill().dropna()
        returns = df_all_prices.pct_change().dropna()

    # --- 功能：下載資料 ---
    st.success(f"✅ 資料載入完成！交易日共 {len(df_all_prices)} 天")
    csv_data = df_all_prices.to_csv().encode('utf-8')
    st.download_button("📥 下載原始價格數據 (CSV)", csv_data, "market_data.csv", "text/csv")

    # --- 還原所有原有的 Tab 分頁 ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 A. 統計特徵", "🔗 B. 相關性矩陣", "💰 C. 投資模擬", "📐 D. 市場模型", "⚖️ E. 效率前緣", "🔮 F. 未來預測"
    ])

    with tab1:
        st.subheader("📊 A. 資產報酬統計")
        stats_df = pd.DataFrame(index=returns.columns)
        stats_df['Ann. Return'] = returns.mean() * 252
        stats_df['Total Return'] = (df_all_prices.iloc[-1] / df_all_prices.iloc[0]) - 1
        stats_df['Ann. Volatility'] = returns.std() * np.sqrt(252)
        stats_df['Skew'] = returns.skew()
        stats_df['Kurt'] = returns.kurt()
        stats_df['JB_p'] = [stats.jarque_bera(returns[c])[1] for c in returns.columns]
        
        # 新增 MDD 指標
        mdd_vals = [calculate_mdd(df_all_prices[c])[0] for c in df_all_prices.columns]
        stats_df['Max Drawdown'] = mdd_vals

        display_df = stats_df.copy()
        for col in ['Ann. Return', 'Total Return', 'Ann. Volatility', 'Max Drawdown']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        display_df['Normality'] = display_df['JB_p'].apply(interpret_jb_test)
        st.dataframe(display_df, use_container_width=True)

        # 直方圖
        cols = st.columns(2)
        for i, asset in enumerate(returns.columns):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(returns[asset].dropna(), bins=40, color='#2980b9', alpha=0.7, density=True)
                st.pyplot(fig)

    with tab2:
        st.subheader("🔗 B. 相關性矩陣")
        st.pyplot(plot_heatmap_matplotlib(returns.corr()), use_container_width=True)

    with tab3:
        st.subheader("💰 C. 投資模擬")
        cum_wealth = (1 + returns).cumprod() * initial_capital
        st.line_chart(cum_wealth)

    with tab4:
        st.subheader("📐 D. 市場模型 (CAPM)")
        mkt = '0050' if '0050' in returns.columns else returns.columns[0]
        res = []
        for stock in [s for s in returns.columns if s != mkt]:
            slope, intercept, r_val, p_val, _ = stats.linregress(returns[mkt], returns[stock])
            res.append({"Asset": stock, "Beta": slope, "R-Squared": r_val**2})
        st.dataframe(pd.DataFrame(res).set_index("Asset"))

    with tab5:
        st.subheader("⚖️ E. 效率前緣")
        mean_v, cov_m = returns.mean()*252, returns.cov()*252
        sim_res = np.zeros((3, num_simulations))
        for i in range(num_simulations):
            w = np.random.random(len(returns.columns)); w /= w.sum()
            pr, pv = np.sum(w*mean_v), np.sqrt(np.dot(w.T, np.dot(cov_m, w)))
            sim_res[:, i] = [pr, pv, (pr-rf)/pv]
        tidx = np.argmax(sim_res[2])
        fig, ax = plt.subplots()
        ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10)
        ax.scatter(sim_res[1, tidx], sim_res[0, tidx], c='red', marker='*', s=200)
        st.pyplot(fig)

    with tab6:
        st.subheader("🔮 F. 未來預測")
        tgt = st.selectbox("選擇預測標的", returns.columns)
        mu_raw, sigma_raw = returns[tgt].mean() * 252, returns[tgt].std() * np.sqrt(252)
        s0 = df_all_prices[tgt].iloc[-1]
        dt = 1/252
        sim_df = pd.DataFrame()
        for x in range(100):
            path = s0 * np.exp(np.cumsum((mu_raw-0.5*sigma_raw**2)*dt + sigma_raw*np.sqrt(dt)*np.random.normal(0,1,forecast_days)))
            sim_df[f's{x}'] = path
        st.line_chart(sim_df)
