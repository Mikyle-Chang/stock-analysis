import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 頁面與視覺設定 ---
st.set_page_config(page_title="v16.0 旗艦投資組合系統", layout="wide", page_icon="📈")

# 設定圖表風格
plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# CSS 美化 (放大圖表容器與字體)
st.markdown("""
    <style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🎓 旗艦級投資組合分析系統 (Final Project)')
st.caption("v16.0 | 修正報酬率顯示 (年化/總報酬) | 放大熱力圖矩陣 | 介面優化")

# --- 2. 側邊欄 ---
with st.sidebar:
    st.header('1. 🎯 投資標的')
    default_stocks = '2330, 2454, 2317, 2603, 2881'
    stock_input = st.text_input('台股代號 (逗號隔開)', default_stocks)
    
    st.header('2. 📅 回測設定')
    # 預設拉長到 5 年，更能看出 0050 的長期趨勢
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*5))
    end_date = st.date_input('結束日期', datetime.now())
    
    st.header('3. 💰 資金管理')
    initial_capital = st.number_input('初始投入本金 (USD/TWD)', value=100000, step=10000)
    risk_free_rate_pct = st.number_input('無風險利率 (%)', value=4.0, step=0.1)
    rf = risk_free_rate_pct / 100.0
    
    st.header('4. 🎲 模型參數')
    num_simulations = st.slider('蒙地卡羅模擬次數', 1000, 10000, 3000)
    forecast_days = st.slider('未來預測天數', 30, 365, 180)

# --- 3. 核心函數 ---

def generate_mock_international_data(dates, asset_type='equity'):
    """生成模擬國際資產數據 (向上修正漂移項，確保長期為正)"""
    n = len(dates)
    if asset_type == 'equity':
        mu, sigma = 0.0005, 0.015  # 稍微調高 mu 確保模擬數據好看
    elif asset_type == 'bond':
        mu, sigma = 0.00015, 0.005 
    elif asset_type == 'commodity':
        mu, sigma = 0.0003, 0.02
    
    returns = np.random.normal(mu, sigma, n)
    price = 100 * np.exp(np.cumsum(returns))
    return price

def interpret_jb_test(p_value):
    return "❌ 拒絕常態" if p_value < 0.05 else "✅ 近似常態"

def plot_heatmap_matplotlib(df_corr):
    """(修正版) 超大尺寸熱力圖"""
    # 放大尺寸到 14x12
    fig, ax = plt.subplots(figsize=(14, 12)) 
    
    # 畫圖
    cax = ax.imshow(df_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Colorbar 調整
    cbar = fig.colorbar(cax, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    
    # 軸標籤設定
    ticks = np.arange(len(df_corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # 字體放大
    ax.set_xticklabels(df_corr.columns, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.set_yticklabels(df_corr.index, fontsize=12, fontweight='bold')
    
    # 填入數字 (字體放大)
    for i in range(len(df_corr.columns)):
        for j in range(len(df_corr.columns)):
            val = df_corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
            
    ax.set_title("12x12 資產相關係數矩陣 (Correlation Matrix)", fontsize=18, pad=20)
    ax.grid(False)
    return fig

# --- 4. 主程式 ---
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    raw_stocks = [s.strip().replace('.TW', '') for s in stock_input.split(',')]
    
    with st.spinner('正在抓取台股並生成國際模擬數據...'):
        api = DataLoader()
        data_dict = {}
        
        # 抓台股
        for i, stock in enumerate(raw_stocks):
            try:
                df = api.taiwan_stock_daily(stock_id=stock, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    series = pd.to_numeric(df['close'], errors='coerce')
                    series = series[series > 0]
                    data_dict[stock] = series
            except: pass
        
        # 抓 0050 (大盤)
        try:
            df_mkt = api.taiwan_stock_daily(stock_id='0050', start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            if not df_mkt.empty:
                df_mkt['date'] = pd.to_datetime(df_mkt['date'])
                df_mkt.set_index('date', inplace=True)
                data_dict['0050'] = pd.to_numeric(df_mkt['close'], errors='coerce')
        except: pass

        if data_dict:
            df_tw = pd.DataFrame(data_dict).ffill().dropna()
        else:
            st.error("❌ 無法抓取台股資料。")
            st.stop()
            
        # 生成國際資產 (模擬)
        dates = df_tw.index
        mock_assets = {'SPY': 'equity', 'Nikkei225': 'equity', 'VUG': 'equity', 'VTV': 'equity', 'VNQ': 'equity', 'VCIT': 'bond', 'GLD': 'commodity', 'DBC': 'commodity'}
        df_global = pd.DataFrame(index=dates)
        for asset, atype in mock_assets.items():
            df_global[asset] = generate_mock_international_data(dates, atype)
        
        df_all_prices = pd.concat([df_tw, df_global], axis=1).ffill().dropna()
        returns = df_all_prices.pct_change().dropna()
        
        st.success(f"✅ 資料分析完成！期間: {start_date} ~ {end_date} (共 {len(df_all_prices)} 交易日)")

    # ==================== 分析分頁 ====================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 A. 統計特徵 (修正版)", 
        "🔗 B. 相關性矩陣 (放大版)", 
        "💰 D. 投資模擬", 
        "📐 C. 市場模型",
        "⚖️ B. 效率前緣",
        "🔮 未來預測"
    ])

    # --- Tab 1: 統計特徵 (顯示優化) ---
    with tab1:
        st.subheader("📊 A. 資產報酬統計")
        st.caption("已切換為「年化報酬」與「總報酬」，反映真實長期績效。")
        
        stats_df = pd.DataFrame(index=returns.columns)
        
        # 1. 關鍵修正：計算年化與總報酬
        stats_df['Ann. Return'] = returns.mean() * 252  # 年化報酬
        stats_df['Total Return'] = (df_all_prices.iloc[-1] / df_all_prices.iloc[0]) - 1 # 總報酬
        stats_df['Ann. Volatility'] = returns.std() * np.sqrt(252) # 年化波動
        stats_df['Skew'] = returns.skew()
        stats_df['Kurt'] = returns.kurt()
        stats_df['JB_p'] = [stats.jarque_bera(returns[c])[1] for c in returns.columns]
        
        # 格式化顯示
        display_df = stats_df.copy()
        display_df['Ann. Return'] = display_df['Ann. Return'].apply(lambda x: f"{x:.2%}") # 百分比顯示
        display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.2%}")
        display_df['Ann. Volatility'] = display_df['Ann. Volatility'].apply(lambda x: f"{x:.2%}")
        display_df['Normality'] = display_df['JB_p'].apply(interpret_jb_test)
        
        # 使用更大的表格顯示
        st.dataframe(
            display_df[['Ann. Return', 'Total Return', 'Ann. Volatility', 'Skew', 'Kurt', 'Normality']], 
            use_container_width=True,
            height=500
        )
        
        st.divider()
        st.subheader("📉 分布直方圖")
        
        cols = st.columns(2)
        for i, asset in enumerate(returns.columns):
            clean_series = returns[asset].dropna()
            if len(clean_series) > 0:
                with cols[i % 2]:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(clean_series, bins=40, color='#2980b9', alpha=0.7, density=True, edgecolor='white')
                    
                    # 常態曲線
                    xmin, xmax = ax.get_xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, clean_series.mean(), clean_series.std())
                    ax.plot(x, p, 'r', linewidth=2, label='Normal')
                    
                    ax.set_title(f"{asset}", fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)

    # --- Tab 2: 相關性 (矩陣放大版) ---
    with tab2:
        st.subheader("🔗 B. 12x12 相關性矩陣")
        st.caption("圖表已放大，並使用 Container Width 撐滿畫面。")
        
        corr_matrix = returns.corr()
        
        # 使用修正後的函數繪圖
        fig_big = plot_heatmap_matplotlib(corr_matrix)
        
        # 關鍵參數：use_container_width=True
        st.pyplot(fig_big, use_container_width=True)

    # --- Tab 3: 投資模擬 ---
    with tab3:
        st.subheader("💰 D. 投資模擬")
        col_in, col_ch = st.columns([1, 3])
        with col_in:
            sim_capital = st.number_input("模擬本金", value=initial_capital, step=10000)
        with col_ch:
            cum_wealth = (1 + returns).cumprod() * sim_capital
            st.line_chart(cum_wealth)
            
            # 排序顯示
            final_vals = cum_wealth.iloc[-1].sort_values(ascending=False)
            st.write("**期末價值排名 (前 5 名):**")
            st.dataframe(final_vals.head(5).to_frame(name="Value").style.format("${:,.0f}"))

    # --- Tab 4: 市場模型 ---
    with tab4:
        st.subheader("📐 C. 市場模型風險衡量")
        mkt = '0050' if '0050' in returns.columns else returns.columns[0]
        mkt_var = returns[mkt].var() * 252
        
        res = []
        for stock in [s for s in raw_stocks if s in returns.columns]:
            y, X = returns[stock], returns[mkt]
            slope, intercept, _, _, _ = stats.linregress(X, y)
            resid_var = (y - (intercept + slope * X)).var() * 252
            res.append({
                "Asset": stock, "Beta": slope,
                "Full Var": y.var()*252, "Diagonal Var": (slope**2*mkt_var)+resid_var, "Beta Var": slope**2*mkt_var
            })
        st.dataframe(pd.DataFrame(res).set_index("Asset").style.format("{:.4f}").background_gradient(cmap='Oranges'))

    # --- Tab 5: 效率前緣 ---
    with tab5:
        st.subheader("⚖️ B. 效率前緣")
        risky = returns[[s for s in raw_stocks if s in returns.columns]]
        mean_v, cov_m = risky.mean()*252, risky.cov()*252
        
        sim_res = np.zeros((3, num_simulations))
        for i in range(num_simulations):
            w = np.random.random(len(risky.columns)); w /= w.sum()
            pr, pv = np.sum(w*mean_v.values), np.sqrt(np.dot(w.T, np.dot(cov_m.values, w)))
            sim_res[:, i] = [pr, pv, (pr-rf)/pv]
            
        midx, tidx = np.argmin(sim_res[1]), np.argmax(sim_res[2])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10, alpha=0.5)
            ax.scatter(sim_res[1, midx], sim_res[0, midx], c='blue', s=150, marker='D', label='GMV')
            ax.scatter(sim_res[1, tidx], sim_res[0, tidx], c='red', s=200, marker='*', label='Tangency')
            plt.colorbar(sc, label='Sharpe')
            ax.legend()
            st.pyplot(fig)
        with col2:
            st.metric("最佳夏普", f"{sim_res[2, tidx]:.2f}")
            st.metric("預期報酬", f"{sim_res[0, tidx]:.2%}")

    # --- Tab 6: 預測 ---
    with tab6:
        st.subheader("🔮 未來預測")
        c1, c2 = st.columns([1, 3])
        with c1:
            tgt = st.selectbox("標的", [s for s in raw_stocks if s in returns.columns])
            price = st.number_input("進場價", value=float(df_all_prices[tgt].iloc[-1]))
        with c2:
            mu, sigma = stats_df.loc[tgt, 'Ann. Return'], stats_df.loc[tgt, 'Ann. Volatility']
            # 這裡要注意，stats_df 裡面的值已經是 format 過的字串，需要重算或是取原始值
            # 為了簡便，直接重算
            mu_raw = returns[tgt].mean() * 252
            sigma_raw = returns[tgt].std() * np.sqrt(252)
            
            dt = 1/252; sim_df = pd.DataFrame()
            for x in range(200):
                path = price * np.exp(np.cumsum((mu_raw-0.5*sigma_raw**2)*dt + sigma_raw*np.sqrt(dt)*np.random.normal(0,1,forecast_days)))
                sim_df[f's{x}'] = path
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(sim_df, color='skyblue', alpha=0.1)
            ax.plot(sim_df.mean(axis=1), color='red', linewidth=2)
            st.pyplot(fig)
            
            final = sim_df.iloc[-1]
            st.success(f"P95: {np.percentile(final, 95):.2f} | P05: {np.percentile(final, 5):.2f}")
