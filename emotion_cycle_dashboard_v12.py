import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import plotly.express as px

# 設定頁面
st.set_page_config(
    layout="wide", 
    page_title="情緒週期獵人 v13.1 - 風控最終版",
    page_icon="🛡️"
)

@st.cache_data
def load_data():
    """載入數據並自動判讀時間範圍"""
    try:
        # 嘗試載入最新版數據
        try:
            df = pd.read_csv('emotion_cycle_ready_v12_final.csv')
            data_version = "v12_final"
        except FileNotFoundError:
            try:
                df = pd.read_csv('emotion_cycle_ready_v12.csv')
                data_version = "v12"
            except:
                st.error("❌ 找不到數據檔案 (emotion_cycle_ready_v12_final.csv)")
                return None, None, []
            
        df['date'] = pd.to_datetime(df['date'])
        
        # 確保必要欄位存在與計算
        df = ensure_required_columns(df)
        
        # 自動判讀可用年份範圍
        available_years = sorted(df['date'].dt.year.unique())
        # 排除 2019 以前的數據以確保品質
        reliable_years = [year for year in available_years if year >= 2019]
        
        return df, data_version, reliable_years
        
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None, []

def ensure_required_columns(df):
    """確保必要欄位存在，並補足技術指標"""
    
    # 基本欄位檢查
    if 'twii_open' not in df.columns and 'twii' in df.columns:
        df['twii_open'] = df['twii']
    
    # 確保有 200 日均線 (大趨勢濾網用)
    if 'MA_200' not in df.columns and 'twii' in df.columns:
        df['MA_200'] = df['twii'].rolling(200, min_periods=1).mean()
        
    # 確保有漲跌幅 (價格確認用)
    if 'twii_ret' not in df.columns and 'twii' in df.columns:
        df['twii_ret'] = df['twii'].pct_change() * 100
    
    # 翻譯週期名稱
    if 'cycle_phase' in df.columns:
        phase_translation = {
            'NORMAL': '正常期',
            'PANIC_START': '恐慌開始', 
            'PANIC_BUILDING': '恐慌加劇',
            'PANIC_PEAK': '恐慌高峰',
            'FATIGUE': '疲乏期',
            'RECOVERY': '復甦期'
        }
        df['cycle_phase'] = df['cycle_phase'].map(phase_translation).fillna(df['cycle_phase'])
        
    # 確保有信號欄位 (若 ETL 沒產出)
    if 'buy_signal_cycle' not in df.columns:
        df['buy_signal_cycle'] = False
        df['signal_confidence'] = 0.0
    
    return df

def extract_hot_news_titles(df, top_n=8):
    """
    修正版：提取「情緒強度 (Ratio) 最高」時期的熱門新聞
    優先顯示恐慌時刻的標題，而非最近日期
    """
    news_column = None
    for col in ['news_titles', 'top_titles']:
        if col in df.columns:
            news_column = col
            break
    
    if news_column is None:
        return []
    
    # 1. 過濾掉沒有新聞的日子
    df_has_news = df[
        pd.notna(df[news_column]) & 
        (df[news_column] != "No data") & 
        (df[news_column] != "No news")
    ].copy()
    
    if df_has_news.empty:
        return []

    # 2. 關鍵修正：按「情緒比率 (ratio)」降序排列，取前 20 個最恐慌的日子
    panic_days = df_has_news.sort_values('ratio', ascending=False).head(20)
    
    # 3. 再按日期排序，讓顯示時符合時間軸
    panic_days = panic_days.sort_values('date')
    
    all_news = []
    
    for _, row in panic_days.iterrows():
        titles_text = row[news_column]
        ratio_val = row['ratio']
        
        daily_titles = titles_text.split('<br>')
        for title in daily_titles:
            # 簡單過濾清洗
            if '[新聞]' in title or '新聞' in title:
                clean_title = title.replace('🔥', '').replace('[新聞]', '').strip()
                if ']' in clean_title:
                    clean_title = clean_title.split(']', 1)[1].strip()
                
                # 只保留夠長的標題
                if len(clean_title) > 8:
                    all_news.append({
                        'date': row['date'].strftime('%Y/%m/%d'), 
                        'ratio': f"{ratio_val:.1f}", # 記錄當下的恐慌指數
                        'title': clean_title[:45] + ('...' if len(clean_title) > 45 else '')
                    })
                    break # 一天只取一則最重要的新聞
    
    return all_news[:top_n]

def generate_time_options(reliable_years):
    """生成時間選項"""
    if not reliable_years:
        return {"全部數據": None}
    
    min_year = min(reliable_years)
    max_year = max(reliable_years)
    
    options = {
        f"全部數據 ({min_year}-{max_year})": None,
        "最近兩年": 730,
        "最近一年": 365,
        "最近半年": 180,
    }
    
    for year in sorted(reliable_years, reverse=True):
        label = f"{year}年"
        if year == 2022: label += " (熊市/空頭)"
        elif year == 2021: label += " (牛市/多頭)"
        options[label] = str(year)
    
    return options

def robust_strategy_backtest(df, hold_days, cost_rate, stop_loss_pct, take_profit_pct):
    """
    修正後的策略回測 - 嚴格遵守 T日決策 T+1日執行
    包含：MA200 濾網、價格回穩確認、動態停損停利
    """
    trades = []
    active_trade = None # 單一部位模式
    
    # 從第 5 天開始遍歷 (確保有足夠歷史數據)
    for i in range(5, len(df) - 1):
        
        # ==========================================
        # 1. 處理現有持倉 (動態出場檢查)
        # ==========================================
        if active_trade:
            # 取得今日數據 (Day i)
            current_date = df.iloc[i]['date']
            day_open = df.iloc[i]['twii_open']
            
            entry_price = active_trade['entry_price']
            sl_price = entry_price * (1 - stop_loss_pct / 100)
            tp_price = entry_price * (1 + take_profit_pct / 100)
            
            # 檢查持有天數
            days_held = (current_date - active_trade['entry_date']).days
            time_exit = days_held >= hold_days
            
            exit_price = None
            exit_reason = None
            
            # A. 檢查是否開盤就跳空跌破停損 (Gap Down)
            if day_open < sl_price:
                exit_price = day_open
                exit_reason = '停損(跳空)'
            
            # B. 檢查盤中停損 (以收盤價模擬，若有 Low 數據可更嚴格)
            elif df.iloc[i]['twii'] < sl_price:
                exit_price = sl_price
                exit_reason = '停損觸發'
                
            # C. 檢查盤中停利
            elif df.iloc[i]['twii'] > tp_price:
                exit_price = tp_price
                exit_reason = '停利達標'
                
            # D. 時間到出場
            elif time_exit:
                exit_price = df.iloc[i]['twii'] # 收盤出場
                exit_reason = '時間到期'
            
            if exit_price:
                gross_ret = (exit_price / entry_price - 1) * 100
                net_ret = gross_ret - cost_rate
                
                trades.append({
                    '信號日期': active_trade['signal_date'],
                    '進場日期': active_trade['entry_date'],
                    '出場日期': current_date,
                    '主要策略': active_trade['strategy'],
                    '進場價格': entry_price,
                    '出場價格': exit_price,
                    '出場原因': exit_reason,
                    '毛報酬': gross_ret,
                    '淨報酬': net_ret,
                    '持有天數': days_held,
                    '信號信心': active_trade['confidence']
                })
                active_trade = None # 清空持倉
                
            continue # 如果有持倉，今天就不再開新倉

        # ==========================================
        # 2. 尋找新買點 (Day i 收盤後決策)
        # ==========================================
        
        # 基礎數據 (全部來自 Day i 或之前，無未來數據)
        curr_ratio = df.iloc[i]['ratio']
        curr_ret = df.iloc[i]['twii_ret']
        curr_close = df.iloc[i]['twii']
        ma_200 = df.iloc[i]['MA_200']
        
        # --- 🛡️ 修正 1：大趨勢濾網 (Regime Filter) ---
        is_bear_market = curr_close < ma_200
        
        # --- 🛡️ 修正 2：價格回穩確認 (Price Support) ---
        is_falling_knife = curr_ret < -0.5 
        
        buy_signals = []
        confidences = []
        
        # [策略 A] 疲乏買點 (PTT 情緒退潮)
        if df.iloc[i]['buy_signal_cycle']:
            # 必須不是正在大跌
            if not is_falling_knife: 
                base_conf = df.iloc[i]['signal_confidence']
                # 熊市打折
                final_conf = base_conf * 0.7 if is_bear_market else base_conf
                
                if final_conf > 0.5:
                    buy_signals.append('FATIGUE')
                    confidences.append(final_conf)
        
        # [策略 B] 傳統恐慌買點 (暴跌反彈)
        if i >= 1:
            prev_ratio = df.iloc[i-1]['ratio']
            prev_ret = df.iloc[i-1]['twii_ret']
            
            # 昨大跌且恐慌 + 今情緒降溫
            if (prev_ratio > 2.5 and prev_ret < -1.0 and curr_ratio < prev_ratio):
                # 熊市濾網：熊市需要極度恐慌才買 (Ratio > 3.0)
                if not is_bear_market or (is_bear_market and prev_ratio > 3.0):
                    if not is_falling_knife:
                        buy_signals.append('TRADITIONAL')
                        confidences.append(0.85)
        
        # [策略 C] 適應性買點 (僅牛市開啟)
        if not is_bear_market:
            if i >= 20:
                vol = df.iloc[i-20:i]['ratio'].std()
                adapt_thresh = 1.5 + vol
                if curr_ratio > adapt_thresh and df.iloc[i-1]['ratio'] > curr_ratio:
                    if not is_falling_knife:
                        buy_signals.append('ADAPTIVE')
                        confidences.append(0.6)

        # ==========================================
        # 3. 執行進場 (準備在 Day i+1 開盤買)
        # ==========================================
        if buy_signals:
            best_idx = np.argmax(confidences)
            chosen_strat = buy_signals[best_idx]
            chosen_conf = confidences[best_idx]
            
            active_trade = {
                'signal_date': df.iloc[i]['date'],
                'entry_date': df.iloc[i+1]['date'], # 明天
                'entry_price': df.iloc[i+1]['twii_open'], # 明天開盤價
                'strategy': chosen_strat,
                'confidence': chosen_conf
            }

    return pd.DataFrame(trades)

def analyze_strategy_performance(trades_df):
    """分析策略績效"""
    if len(trades_df) == 0:
        return {}
    
    returns = trades_df['淨報酬']
    
    performance = {
        '交易次數': len(trades_df),
        '總報酬': f"{returns.sum():.2f}%",
        '勝率': f"{(returns > 0).mean()*100:.1f}%", 
        '平均報酬': f"{returns.mean():.2f}%",
        '最大獲利': f"{returns.max():.2f}%",
        '最大虧損': f"{returns.min():.2f}%",
        '盈虧比': f"{abs(returns[returns>0].mean() / returns[returns<0].mean()):.2f}" if len(returns[returns<0]) > 0 else "Inf"
    }
    
    return performance

def create_main_chart(df, trades_df):
    """創建互動圖表"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("📈 台股指數 (含 MA200 趨勢線)", "🌊 情緒強度"),
        vertical_spacing=0.05
    )
    
    # 1. 價格線
    fig.add_trace(go.Scatter(x=df['date'], y=df['twii'], name="加權指數", line=dict(color='#2E86C1')), row=1, col=1)
    
    # 2. MA200 (趨勢濾網視覺化)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA_200'], name="200日均線 (牛熊分界)", 
                            line=dict(color='#888888', dash='dot', width=1)), row=1, col=1)

    # 3. 交易點位
    if not trades_df.empty:
        fig.add_trace(go.Scatter(
            x=trades_df['進場日期'], y=trades_df['進場價格'],
            mode='markers', marker=dict(symbol='triangle-up', color='green', size=10),
            name='買進', text=trades_df['主要策略']
        ), row=1, col=1)
        
        # 區分獲利與虧損出場
        profit_trades = trades_df[trades_df['淨報酬'] > 0]
        loss_trades = trades_df[trades_df['淨報酬'] <= 0]
        
        if not profit_trades.empty:
            fig.add_trace(go.Scatter(
                x=profit_trades['出場日期'], y=profit_trades['出場價格'],
                mode='markers', marker=dict(symbol='triangle-down', color='red', size=8),
                name='獲利出場'
            ), row=1, col=1)
            
        if not loss_trades.empty:
            fig.add_trace(go.Scatter(
                x=loss_trades['出場日期'], y=loss_trades['出場價格'],
                mode='markers', marker=dict(symbol='x', color='black', size=8),
                name='停損出場'
            ), row=1, col=1)

    # 4. 情緒指標
    fig.add_trace(go.Scatter(x=df['date'], y=df['ratio'], name="情緒Ratio", 
                            line=dict(color='#E74C3C')), row=2, col=1)
    
    # 閥值線
    if 'panic_threshold' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['panic_threshold'], name="恐慌線",
                                line=dict(color='gray', dash='dot', width=1)), row=2, col=1)

    fig.update_layout(height=600, hovermode="x unified", template="plotly_white")
    return fig

# ==================== 主程式 ====================

data_result = load_data()
if data_result[0] is None:
    st.stop()

df_raw, data_version, reliable_years = data_result

# --- Sidebar ---
st.sidebar.title("🌊 情緒週期獵人 v13.1")
st.sidebar.caption("🛡️ 風控最終版 (Regime Filter + News Fix)")

# 時間選擇
time_options = generate_time_options(reliable_years)
selected_period = st.sidebar.selectbox("📅 時間區間", list(time_options.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 策略參數")

# 風控參數
hold_days = st.sidebar.slider("最大持有天數", 5, 40, 20)
stop_loss = st.sidebar.slider("🛑 停損比例 (%)", 2.0, 15.0, 7.0, 0.5)
take_profit = st.sidebar.slider("💰 停利比例 (%)", 5.0, 50.0, 15.0, 1.0)
cost_rate = st.sidebar.slider("交易成本 (%)", 0.0, 1.0, 0.2)

# --- 數據過濾 ---
if time_options[selected_period] is None:
    df = df_raw[df_raw['date'].dt.year.isin(reliable_years)].copy()
elif isinstance(time_options[selected_period], int): # 天數
    cutoff = df_raw['date'].max() - pd.Timedelta(days=time_options[selected_period])
    df = df_raw[df_raw['date'] >= cutoff].copy()
else: # 特定年份
    year = int(time_options[selected_period])
    df = df_raw[df_raw['date'].dt.year == year].copy()

df = df.reset_index(drop=True)

# --- 執行回測 ---
trades_df = robust_strategy_backtest(df, hold_days, cost_rate, stop_loss, take_profit)
perf = analyze_strategy_performance(trades_df)

# --- 顯示結果 ---
st.title("🛡️ 情緒週期獵人 - 風控最終版")

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("交易次數", perf.get('交易次數', 0))
with col2: st.metric("總報酬", perf.get('總報酬', '0%'))
with col3: st.metric("勝率", perf.get('勝率', '0%'))
with col4: st.metric("最大虧損", perf.get('最大虧損', '0%'))
with col5: st.metric("盈虧比", perf.get('盈虧比', '0'))

st.plotly_chart(create_main_chart(df, trades_df), use_container_width=True)

# --- 📰 (NEW) 恐慌時刻頭條新聞 ---
st.markdown("---")
st.subheader("😱 歷史恐慌時刻頭條 (Top Fear News)")

# 自動抓取區間內 Ratio 最高的日子
hot_news = extract_hot_news_titles(df, 8) 

if hot_news:
    col1, col2 = st.columns(2)
    for i, news in enumerate(hot_news):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            # 視覺化情緒強度
            ratio_float = float(news['ratio'])
            ratio_color = "#D62728" if ratio_float > 3.0 else "#FF7F0E"
            
            st.markdown(
                f"<small>{news['date']}</small> "
                f"<span style='background-color:{ratio_color}; color:white; padding:2px 6px; border-radius:4px; font-size:0.8em'>Ratio: {news['ratio']}</span><br>"
                f"**{news['title']}**", 
                unsafe_allow_html=True
            )
else:
    st.info("⚠️ 此區間無足夠新聞數據，或 ETL 階段未產生新聞標題。")

# --- 詳細交易列表 ---
st.markdown("---")
if not trades_df.empty:
    with st.expander("📝 查看詳細交易記錄", expanded=True):
        show_df = trades_df[['進場日期', '主要策略', '進場價格', '出場價格', '出場原因', '淨報酬', '持有天數']].copy()
        
        show_df['進場價格'] = show_df['進場價格'].apply(lambda x: f"{x:.0f}")
        show_df['出場價格'] = show_df['出場價格'].apply(lambda x: f"{x:.0f}")
        show_df['淨報酬'] = show_df['淨報酬'].apply(lambda x: f"{x:.2f}%")
        
        def color_ret(val):
            try:
                val_float = float(val.strip('%'))
                color = '#ffcccc' if val_float < 0 else '#ccffcc'
                return f'background-color: {color}'
            except:
                return ''
        
        st.dataframe(show_df.style.applymap(color_ret, subset=['淨報酬']), use_container_width=True)
else:
    st.warning("⚠️ 此區間無符合策略與風控條件的交易信號")