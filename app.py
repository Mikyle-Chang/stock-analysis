import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from FinMind.data import DataLoader
from datetime import datetime, timedelta

# --- 1. 頁面設定與工具函數 ---
st.set_page_config(page_title="台股投資組合分析", layout="wide", page_icon="📈")

# 設定中文字體與風格
plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def calculate_mdd(series):
    """計算最大回撤邏輯"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min(), drawdown

# --- 2. 側邊欄與參數設定 ---
with st.sidebar:
    st.header('🎯 投資標的設定')
    default_stocks = '2330, 2454, 2317, 2603, 2881'
    stock_input = st.text_input('台股代號 (請用逗號隔開)', default_stocks)
    
    st.header('📅 時間與資金')
    start_date = st.date_input('開始日期', datetime.now() - timedelta(days=365*3))
    end_date = st.date_input('結束日期', datetime.now())
    initial_capital = st.number_input('初始本金 (TWD)', value=100000)
    rf = st.number_input('無風險利率 (%)', value=2.0) / 100
    
    st.header('🎲 模擬參數')
    num_simulations = st.slider('蒙地卡羅模擬次數', 1000, 5000, 2000)
    forecast_days = st.slider('未來預測天數', 30, 365, 120)

# --- 3. 資料抓取模組 ---
if st.sidebar.button('🚀 開始執行全方位分析', type="primary"):
    raw_stocks = [s.strip() for s in stock_input.split(',')]
    # 確保包含 0050 作為基準
    fetch_list = list(set(raw_stocks + ['0050']))
    
    with st.spinner('正在從 FinMind 抓取真實數據...'):
        api = DataLoader()
        data_dict = {}
        for stock in fetch_list:
            try:
                df = api.taiwan_stock_daily(stock_id=stock, 
                                            start_date=start_date.strftime('%Y-%m-%d'), 
                                            end_date=end_date.strftime('%Y-%m-%d'))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    data_dict[stock] = df.set_index('date')['close']
            except: pass
        
        if not data_dict:
            st.error("無法取得資料，請檢查網路或代號。")
            st.stop()
            
        df_prices = pd.DataFrame(data_dict).ffill().dropna()
        returns = df_prices.pct_change().dropna()

    # --- 4. 數據計算與分頁 ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 統計分析", "🔗 相關性與風險", "⚖️ 組合優化", "🔮 未來預測"])

    # Tab 1: 統計指標 (含 MDD)
    with tab1:
        st.subheader("📋 歷史表現統計")
        stats_df = pd.DataFrame(index=df_prices.columns)
        stats_df['總報酬率'] = (df_prices.iloc[-1] / df_prices.iloc[0] - 1)
        stats_df['年化報酬率'] = returns.mean() * 252
        stats_df['年化波動率'] = returns.std() * np.sqrt(252)
        stats_df['夏普比率'] = (stats_df['年化報酬率'] - rf) / stats_df['年化波動率']
        
        mdd_list = []
        for col in df_prices.columns:
            mdd_val, _ = calculate_mdd(df_prices[col])
            mdd_list.append(mdd_val)
        stats_df['最大回撤 (MDD)'] = mdd_list

        st.dataframe(stats_df.style.format("{:.2%}"), use_container_width=True)

    # Tab 2: 相關性矩陣
    with tab2:
        st.subheader("🔗 標的相關性矩陣")
        corr = returns.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns); ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center')
        st.pyplot(fig)

    # Tab 3: 效率前緣
    with tab3:
        st.subheader("⚖️ 馬可維茲投資組合優化 (蒙地卡羅)")
        # 排除 0050 後的個股組合
        risky_assets = [s for s in raw_stocks if s in returns.columns]
        r_mean = returns[risky_assets].mean() * 252
        r_cov = returns[risky_assets].cov() * 252
        
        results = np.zeros((3, num_simulations))
        for i in range(num_simulations):
            w = np.random.random(len(risky_assets))
            w /= np.sum(w)
            p_ret = np.sum(w * r_mean)
            p_std = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
            results[0,i] = p_ret
            results[1,i] = p_std
            results[2,i] = (p_ret - rf) / p_std
        
        best_idx = np.argmax(results[2])
        st.write(f"最佳夏普比率組合：預期報酬 {results[0,best_idx]:.2%}, 風險 {results[1,best_idx]:.2%}")
        
        fig, ax = plt.subplots()
        ax.scatter(results[1], results[0], c=results[2], cmap='viridis', s=5)
        ax.scatter(results[1, best_idx], results[0, best_idx], color='red', marker='*', s=200, label='Best Sharpe')
        ax.set_xlabel("年化波動率 (風險)"); ax.set_ylabel("預期報酬率")
        st.pyplot(fig)

    # Tab 4: 未來預測
    with tab4:
        st.subheader("🔮 股價隨機漫步模擬 (GBM)")
        target = st.selectbox("選擇預測標的", risky_assets)
        s0 = df_prices[target].iloc[-1]
        mu = returns[target].mean() * 252
        sigma = returns[target].std() * np.sqrt(252)
        
        dt = 1/252
        paths = np.zeros((forecast_days, 100))
        for i in range(100):
            prices = [s0]
            for _ in range(forecast_days-1):
                prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()))
            paths[:, i] = prices
        
        st.line_chart(pd.DataFrame(paths))
        st.write(f"預測 {forecast_days} 天後平均價格：{np.mean(paths[-1]):.2f}")
