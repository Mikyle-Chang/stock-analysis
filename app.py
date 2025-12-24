import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
    unique_tw = list(set(tickers_tw + ['0050']))
    unique_us = list(set(tickers_us + ['SPY']))
    
    for s in unique_tw:
        if not s: continue
        try:
            ticker = f"{s}.TW"
            yf_obj = yf.Ticker(ticker)
            df = yf_obj.history(start=start, end=end, interval="1d", auto_adjust=True)
            if not df.empty:
                data_dict[s] = df['Close']
        except:
            st.sidebar.warning(f"台股 {s} 抓取嘗試失敗")

    for s in unique_us:
        if not s: continue
        try:
            yf_obj = yf.Ticker(s)
            df = yf_obj.history(start=start, end=end, interval="1d", auto_adjust=True)
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
    num_simulations = st.slider('蒙地卡羅次數', 1000, 5000, 2000)
    forecast_len = st.slider('預測天數', 30, 365, 180)

# --- 5. 主程式執行 ---

# 1. 初始化 Session State 狀態（防止拉桿觸發重新整理導致畫面消失）
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# 2. 點擊按鈕後，將狀態設為 True
if st.sidebar.button('🚀 啟動全方位分析', type="primary"):
    st.session_state.analysis_started = True

# 3. 根據狀態決定是否顯示分析內容
if st.session_state.analysis_started:
    tw_list = [x.strip() for x in tw_in.split(',') if x.strip()]
    us_list = [x.strip().upper() for x in us_in.split(',') if x.strip()]
    
    with st.spinner('正在從 Yahoo Finance 節點抓取全球複權數據...'):
        raw_data = fetch_stock_data(tw_list, us_list, start_date, end_date)
        
        if not raw_data:
            st.error("❌ 所有來源均連線失敗。")
            st.stop()
            
        df_prices = pd.DataFrame(raw_data).ffill().dropna()
        returns = df_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    st.success(f"✅ 成功載入 {len(df_prices.columns)} 檔資產數據！")
    st.download_button("📥 下載調整後數據 (CSV)", df_prices.to_csv().encode('utf-8'), "data.csv")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 統計", "🔗 相關性", "💰 模擬", "📐 市場模型", "⚖️ 效率前緣", "🔮 預測", "🚨 壓力測試"])

    with tab1:
        st.subheader("📋 統計特徵")
        res_df = pd.DataFrame(index=returns.columns)
        total_days = (df_prices.index[-1] - df_prices.index[0]).days
        years = max(total_days / 365.25, 0.1) 
        
        res_df['年化報酬'] = (df_prices.iloc[-1] / df_prices.iloc[0]) ** (1 / years) - 1
        res_df['年化波動'] = returns.std() * np.sqrt(252)
        res_df['夏普比率'] = (res_df['年化報酬'] - rf_rate) / res_df['年化波動']
        res_df['最大回撤'] = [calculate_mdd(df_prices[c])[0] for c in df_prices.columns]
        
        res_df['符合常態'] = [("✅ 是" if stats.jarque_bera(returns[c])[1] > 0.05 else "❌ 否") for c in returns.columns]
        
        numeric_cols = ['年化報酬', '年化波動', '夏普比率', '最大回撤']
        st.dataframe(res_df.style.format({c: "{:.2%}" for c in numeric_cols}), use_container_width=True)
        
        cols = st.columns(2)
        for i, col in enumerate(returns.columns):
            with cols[i%2]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(returns[col], bins=40, density=True, alpha=0.7, color='steelblue')
                ax.set_title(f"{col} Distribution of Returns")
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
        beta_data = []
        for s in [c for c in returns.columns if c not in ['0050', 'SPY']]:
            if s.isdigit() and '0050' in returns.columns:
                mkt_ref = '0050'
            elif not s.isdigit() and 'SPY' in returns.columns:
                mkt_ref = 'SPY'
            else: continue
            common_df = pd.concat([returns[mkt_ref], returns[s]], axis=1).dropna()
            if len(common_df) > 10:
                slope, _, r_val, _, _ = stats.linregress(common_df.iloc[:,0], common_df.iloc[:,1])
                beta_data.append({"Asset": s, "Benchmark": mkt_ref, "Beta": slope, "R2": r_val**2})
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
            all_weights[i, 🙂 = w
            p_r = np.sum(w * r_mean)
            p_v = np.sqrt(np.dot(w.T, np.dot(r_cov, w)))
            sim_res[:, i] = [p_r, p_v, (p_r - rf_rate) / p_v]
        
        tidx = np.argmax(sim_res[2])
        best_weights = all_weights[tidx, 🙂
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write("效率前緣分佈圖")
            fig, ax = plt.subplots(figsize=(10, 6))
            sc = ax.scatter(sim_res[1], sim_res[0], c=sim_res[2], cmap='viridis', s=10, alpha=0.5)
            ax.scatter(sim_res[1, tidx], sim_res[0, tidx], color='red', marker='*', s=200, label='MSR')
            ax.set_xlabel("Risk"); ax.set_ylabel("Exp. Ret.")
            plt.colorbar(sc, label='sharp ratio')
            st.pyplot(fig)

        with col2:
            st.write("最佳資產配置比例")
            df_weights = pd.DataFrame({'資產': returns.columns, '比例': best_weights * 100})
            df_weights = df_weights.sort_values(by='比例', ascending=False)
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(df_weights['比例'], labels=df_weights['資產'], autopct='%1.1f%%', startangle=140)
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
            st.dataframe(df_weights.style.format({'比例': '{:.2f}%'}))

    # --- TAB 6 修改：僅針對 TAB5 最佳組合進行預測 ---
    with tab6:
        st.subheader("🔮 最佳投資組合未來預測 (GBM)")
        
        # 1. 計算最佳組合的歷史報酬率序列
        port_returns_series = (returns * best_weights).sum(axis=1)
        
        # 2. 取得組合的年化參數
        mu_p = port_returns_series.mean() * 252
        sigma_p = port_returns_series.std() * np.sqrt(252)
        s0 = initial_cap  # 模擬起點設定為初始本金
        dt = 1/252
        
        # 3. 執行 GBM 模擬 (維持原有的 50 條路徑邏輯)
        sim_paths = np.zeros((forecast_len, 50))
        sim_paths[0] = s0
        
        drift = (mu_p - 0.5 * sigma_p**2) * dt
        shock = sigma_p * np.sqrt(dt)
        
        for t in range(1, forecast_len):
            z = np.random.normal(0, 1, 50)
            sim_paths[t] = sim_paths[t-1] * np.exp(drift + shock * z)
            
        # 4. 繪製圖表
        st.line_chart(sim_paths)
        
        # 5. 輸出組合預測基準資訊
        st.write(f"預測基準：Tab 5 計算之最佳夏普組合 (MSR)")
        st.info(f"組合年化預期報酬: {mu_p:.2%}, 年化波動率 (風險): {sigma_p:.2%}")
        
    # --- TAB 7: 壓力測試 ---
        with tab7:
            st.subheader("🚨 投資組合壓力測試 (Stress Test)")
            
            # 1. 計算組合的加權 Beta (反映組合對市場的敏感度)
            # 這裡從你 TAB 4 的 beta_data 提取資料
            if len(beta_data) > 0:
                df_beta = pd.DataFrame(beta_data)
                # 建立權重字典方便查詢
                weight_dict = dict(zip(returns.columns, best_weights))
                # 計算組合 Beta = Σ (權重 * 個股 Beta)
                df_beta['Weighted Beta'] = df_beta.apply(lambda x: x['Beta'] * weight_dict.get(x['Asset'], 0), axis=1)
                port_beta = df_beta['Weighted Beta'].sum()
            else:
                port_beta = 1.0 # 預設值
                
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write("*自定義市場衝擊預測*")
                mkt_shock = st.slider("假設大盤(市場基準)下跌 (%)", -50, 0, -10)
                
                # 預估損失 = 本金 * 市場跌幅 * 組合 Beta
                est_loss_pct = (mkt_shock / 100) * port_beta
                est_loss_amt = initial_cap * est_loss_pct
                
                st.metric("預估組合跌幅", f"{est_loss_pct:.2%}", delta=f"{est_loss_pct:.2%}")
                st.metric("預估損失金額", f"${est_loss_amt:,.0f}")
                
            with col2:
                st.write("*歷史極端情境模擬*")
                scenarios = {
                    "2008 金融海嘯 (假設大盤 -20%)": -0.20,
                    "2020 疫情崩盤 (假設大盤 -15%)": -0.15,
                    "2022 升息縮表 (假設大盤 -10%)": -0.10,
                    "微幅修正 (假設大盤 -5%)": -0.05
                }
                
                scene_data = []
                for name, shock in scenarios.items():
                    loss_pct = shock * port_beta
                    scene_data.append({
                        "情境": name,
                        "大盤跌幅": f"{shock:.0%}",
                        "組合預估跌幅": f"{loss_pct:.2%}",
                        "預估損失金額": f"${initial_cap * loss_pct:,.0f}"
                    })
                
                st.table(pd.DataFrame(scene_data))
    
            st.info(f"💡 註：目前組合的加權 Beta 為 **{port_beta:.2f}**。這代表當大盤下跌 1% 時，預計你的組合會隨之變動 {abs(port_beta):.2f}%。")
