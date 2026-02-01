import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date

# ---------------------------------------------------------
# 版本資訊 (隱藏於程式碼)
# Version: v13.5 - UI Enhanced Edition
# Update: 優化側邊欄選單 (分層篩選) + 標籤防重疊 + 暖身期修正
# ---------------------------------------------------------

# 設定頁面
st.set_page_config(
    layout="wide", 
    page_title="PTT股板情緒觀測站",
    page_icon="🌊"
)

@st.cache_data
def load_data():
    """載入數據並自動判讀時間範圍 (含暖身期過濾)"""
    try:
        try:
            df = pd.read_csv('emotion_cycle_ready_v12_final.csv')
        except FileNotFoundError:
            try:
                df = pd.read_csv('emotion_cycle_ready_v12.csv')
            except:
                st.error("❌ 找不到數據檔案 (emotion_cycle_ready_v12_final.csv)")
                return None, None
            
        df['date'] = pd.to_datetime(df['date'])
        
        # 確保數據按時間排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 🔥 暖身期過濾：排除前 180 天以確保指標準確
        warmup_days = 180
        if len(df) > warmup_days:
            df = df.iloc[warmup_days:].reset_index(drop=True)
        
        df = ensure_required_columns(df)
        
        available_years = sorted(df['date'].dt.year.unique())
        reliable_years = [year for year in available_years if year >= 2019]
        
        return df, reliable_years
        
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, []

def ensure_required_columns(df):
    if 'twii_open' not in df.columns and 'twii' in df.columns:
        df['twii_open'] = df['twii']
    if 'MA_200' not in df.columns and 'twii' in df.columns:
        df['MA_200'] = df['twii'].rolling(200, min_periods=1).mean()
    if 'twii_ret' not in df.columns and 'twii' in df.columns:
        df['twii_ret'] = df['twii'].pct_change() * 100
    if 'buy_signal_cycle' not in df.columns:
        df['buy_signal_cycle'] = False
        df['signal_confidence'] = 0.0
    return df

def extract_hot_news_titles(df, top_n=20):
    """
    修正版：提取情緒強度最高時期的熱門新聞
    策略：
    1. 找出全區間 Ratio 最高的 20 個日子
    2. 依照「日期倒序 (新 -> 舊)」排列，讓最近發生的事情排在最上面
    3. 取出前 top_n 筆 (預設 20)
    """
    news_column = next((col for col in ['news_titles', 'top_titles'] if col in df.columns), None)
    if not news_column: return []
    
    # 1. 過濾出有新聞的日子
    df_has_news = df[pd.notna(df[news_column]) & (df[news_column] != "No data") & (df[news_column] != "No news")].copy()
    if df_has_news.empty: return []

    # 2. 取出「情緒最強」的前 20 天 (這是我們的候選名單)
    # 我們希望這 20 個最恐慌的日子都能被顯示，或者至少顯示當中最新的
    target_count = 20
    panic_days = df_has_news.sort_values('ratio', ascending=False).head(target_count)
    
    # 3. 🔥 關鍵修正：改為「日期倒序」排列
    # 這樣 2026/01 的新聞會排在 2025/08 前面
    panic_days = panic_days.sort_values('date', ascending=False)
    
    all_news = []
    
    for _, row in panic_days.iterrows():
        titles_text = row[news_column]
        daily_titles = titles_text.split('<br>')
        for title in daily_titles:
            # 清洗標題
            if '[新聞]' in title or '新聞' in title:
                clean_title = title.replace('🔥', '').replace('[新聞]', '').strip()
                if ']' in clean_title:
                    clean_title = clean_title.split(']', 1)[1].strip()
                
                # 長度過濾
                if len(clean_title) > 8:
                    all_news.append({
                        'date': row['date'].strftime('%Y/%m/%d'), 
                        'ratio': f"{row['ratio']:.1f}", 
                        'title': clean_title[:45] + ('...' if len(clean_title) > 45 else '')
                    })
                    break # 一天只取一則，避免洗版
        
        if len(all_news) >= top_n:
            break
    
    return all_news
    news_column = next((col for col in ['news_titles', 'top_titles'] if col in df.columns), None)
    if not news_column: return []
    
    df_has_news = df[pd.notna(df[news_column]) & (df[news_column] != "No data") & (df[news_column] != "No news")].copy()
    if df_has_news.empty: return []

    panic_days = df_has_news.sort_values('ratio', ascending=False).head(20).sort_values('date')
    
    all_news = []
    for _, row in panic_days.iterrows():
        titles_text = row[news_column]
        daily_titles = titles_text.split('<br>')
        for title in daily_titles:
            if '[新聞]' in title or '新聞' in title:
                clean_title = title.replace('🔥', '').replace('[新聞]', '').strip()
                if ']' in clean_title: clean_title = clean_title.split(']', 1)[1].strip()
                if len(clean_title) > 8:
                    all_news.append({
                        'date': row['date'].strftime('%Y/%m/%d'), 
                        'ratio': f"{row['ratio']:.1f}", 
                        'title': clean_title[:45] + ('...' if len(clean_title) > 45 else '')
                    })
                    break 
    return all_news[:top_n]

# ---------------------------------------------------------
# 🔥 新增功能：側邊欄分層篩選邏輯
# ---------------------------------------------------------
def render_sidebar_filters(df_raw, reliable_years):
    """渲染優化後的側邊欄並返回過濾後的 DataFrame"""
    st.sidebar.title("🌊 PTT股板情緒觀測站")
    st.sidebar.caption("PTT Stock 版大數據分析")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 觀測時間設定")
    
    # 1. 第一層：選擇模式
    mode = st.sidebar.radio(
        "選擇模式", 
        ["🚀 近期趨勢", "🗓️ 年度歷史", "🔍 自訂範圍"], 
        horizontal=True,
        label_visibility="collapsed"
    )
    
    df_filtered = df_raw.copy()
    max_date = df_raw['date'].max()
    min_date = df_raw['date'].min()
    
    # 2. 第二層：根據模式顯示不同選項
    if mode == "🚀 近期趨勢":
        period_map = {
            "最近一年": 365,
            "最近半年": 180,
            "最近一季": 90,
            "最近兩年": 730
        }
        selected_period = st.sidebar.selectbox("選擇區間", list(period_map.keys()), index=0)
        days = period_map[selected_period]
        cutoff = max_date - pd.Timedelta(days=days)
        df_filtered = df_raw[df_raw['date'] >= cutoff]
        
    elif mode == "🗓️ 年度歷史":
        # 製作年度選項 (包含牛熊市標記)
        year_options = {}
        for y in sorted(reliable_years, reverse=True):
            label = f"{y}年"
            if y == 2022: label += " (🐻熊市)"
            elif y == 2021: label += " (🐂牛市)"
            year_options[label] = y
        
        # 加入「全部」選項
        year_options[f"全部數據 ({min(reliable_years)}-{max(reliable_years)})"] = "ALL"
        
        selected_year_label = st.sidebar.selectbox("選擇年份", list(year_options.keys()))
        selected_year = year_options[selected_year_label]
        
        if selected_year != "ALL":
            df_filtered = df_raw[df_raw['date'].dt.year == selected_year]
            
    elif mode == "🔍 自訂範圍":
        # 日期選擇器
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input("開始", value=max_date - pd.Timedelta(days=180), min_value=min_date, max_value=max_date)
        end_date = col2.date_input("結束", value=max_date, min_value=min_date, max_value=max_date)
        
        if start_date <= end_date:
            df_filtered = df_raw[
                (df_raw['date'].dt.date >= start_date) & 
                (df_raw['date'].dt.date <= end_date)
            ]
        else:
            st.sidebar.error("開始日期不能晚於結束日期")
            
    return df_filtered.reset_index(drop=True)

def robust_strategy_backtest(df, hold_days, cost_rate, stop_loss_pct, take_profit_pct):
    trades = []
    active_trade = None
    
    for i in range(5, len(df) - 1):
        if active_trade:
            current_date = df.iloc[i]['date']
            day_open = df.iloc[i]['twii_open']
            entry_price = active_trade['entry_price']
            sl_price = entry_price * (1 - stop_loss_pct / 100)
            tp_price = entry_price * (1 + take_profit_pct / 100)
            days_held = (current_date - active_trade['entry_date']).days
            
            exit_price = None
            exit_reason = None
            
            if day_open < sl_price:
                exit_price = day_open
                exit_reason = '停損(跳空)'
            elif df.iloc[i]['twii'] < sl_price:
                exit_price = sl_price
                exit_reason = '停損觸發'
            elif df.iloc[i]['twii'] > tp_price:
                exit_price = tp_price
                exit_reason = '停利達標'
            elif days_held >= hold_days:
                exit_price = df.iloc[i]['twii']
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
                    '持有天數': days_held
                })
                active_trade = None
            continue

        curr_ratio = df.iloc[i]['ratio']
        curr_ret = df.iloc[i]['twii_ret']
        curr_close = df.iloc[i]['twii']
        ma_200 = df.iloc[i]['MA_200']
        is_bear_market = curr_close < ma_200
        is_falling_knife = curr_ret < -0.5 
        
        buy_signals = []
        confidences = []
        
        if df.iloc[i]['buy_signal_cycle'] and not is_falling_knife:
            base_conf = df.iloc[i]['signal_confidence']
            final_conf = base_conf * 0.7 if is_bear_market else base_conf
            if final_conf > 0.5:
                buy_signals.append('FATIGUE')
                confidences.append(final_conf)
        
        if i >= 1:
            prev_ratio = df.iloc[i-1]['ratio']
            if (prev_ratio > 2.5 and df.iloc[i-1]['twii_ret'] < -1.0 and curr_ratio < prev_ratio):
                if not is_bear_market or (is_bear_market and prev_ratio > 3.0):
                    if not is_falling_knife:
                        buy_signals.append('TRADITIONAL')
                        confidences.append(0.85)

        if buy_signals:
            best_idx = np.argmax(confidences)
            active_trade = {
                'signal_date': df.iloc[i]['date'],
                'entry_date': df.iloc[i+1]['date'],
                'entry_price': df.iloc[i+1]['twii_open'],
                'strategy': buy_signals[best_idx],
                'confidence': confidences[best_idx]
            }

    return pd.DataFrame(trades)

def create_main_chart(df, trades_df=None, show_trades=False):
    """創建互動圖表 (修正：標籤防重疊邏輯改為強度優先)"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4], 
        subplot_titles=("📈 台股指數 (TWII)", "🔥 PTT股板情緒強度 (Ratio)"),
        vertical_spacing=0.08
    )
    
    # 1. 價格線
    fig.add_trace(go.Scatter(x=df['date'], y=df['twii'], name="加權指數", 
                            line=dict(color='#2E86C1', width=1.5)), row=1, col=1)
    
    # 2. MA200
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA_200'], name="200日均線", 
                            line=dict(color='#999999', dash='dot', width=1)), row=1, col=1)

    # 3. 交易點位
    if show_trades and trades_df is not None and not trades_df.empty:
        fig.add_trace(go.Scatter(
            x=trades_df['進場日期'], y=trades_df['進場價格'],
            mode='markers', marker=dict(symbol='triangle-up', color='#00CC96', size=10),
            name='策略買進'
        ), row=1, col=1)
        
        profit_trades = trades_df[trades_df['淨報酬'] > 0]
        if not profit_trades.empty:
            fig.add_trace(go.Scatter(
                x=profit_trades['出場日期'], y=profit_trades['出場價格'],
                mode='markers', marker=dict(symbol='triangle-down', color='#EF553B', size=8),
                name='獲利出場'
            ), row=1, col=1)

    # 4. 情緒指標 - 基礎線
    fig.add_trace(go.Scatter(x=df['date'], y=df['ratio'], name="情緒Ratio", 
                            line=dict(color='#E74C3C', width=1.5)), row=2, col=1)
    
    # 5. 恐慌警戒線與標籤優化
    if 'panic_threshold' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['panic_threshold'], name="動態恐慌線",
                                line=dict(color='gray', dash='dot', width=1)), row=2, col=1)
        
        panic_mask = df['ratio'] > df['panic_threshold']
        panic_df = df[panic_mask]
        
        if not panic_df.empty:
            # A. 紅色標記
            fig.add_trace(go.Scatter(
                x=panic_df['date'], 
                y=panic_df['ratio'],
                mode='markers',
                marker=dict(color='#D62728', size=6),
                name='過熱/恐慌區',
                hoverinfo='skip'
            ), row=2, col=1)
            
            # B. 標籤優化：強度優先 (Highest-First)
            min_days_gap = 12
            
            peaks = []
            for i in range(1, len(df)-1):
                if panic_mask.iloc[i]: 
                    curr = df.iloc[i]['ratio']
                    prev = df.iloc[i-1]['ratio']
                    next_val = df.iloc[i+1]['ratio']
                    
                    if curr > prev and curr > next_val:
                        peaks.append({
                            'date': df.iloc[i]['date'],
                            'val': curr
                        })
            
            peaks.sort(key=lambda x: x['val'], reverse=True)
            
            drawn_dates = []
            for p in peaks:
                p_date = p['date']
                p_val = p['val']
                
                is_colliding = False
                for existing_date in drawn_dates:
                    if abs((p_date - existing_date).days) < min_days_gap:
                        is_colliding = True
                        break
                
                if not is_colliding:
                    drawn_dates.append(p_date)
                    fig.add_annotation(
                        x=p_date,
                        y=p_val,
                        text=p_date.strftime('%m/%d'),
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-30, 
                        font=dict(color='#D62728', size=11, family="Arial"),
                        row=2, col=1
                    )

    fig.update_layout(height=650, hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ==================== 主程式 ====================

data_result = load_data()
if data_result[0] is None:
    st.stop()

df_raw, reliable_years = data_result

# --- 🔥 使用新的側邊欄過濾功能 ---
df = render_sidebar_filters(df_raw, reliable_years)

# --- 主標題 ---
st.title("🌊 PTT股板情緒觀測站")

# --- 頁面分流 (Tabs) ---
tab1, tab2 = st.tabs(["📊 市場情緒觀測", "🧪 策略回測實驗室"])

# ==================== Tab 1: 觀測模式 ====================
with tab1:
    st.markdown("### 📈 市場走勢與 PTT 情緒強度")
    st.caption("情緒強度 (Ratio) 紅色標記代表 PTT 討論過度恐慌，通常為市場潛在轉折點。")
    
    if not df.empty:
        st.plotly_chart(create_main_chart(df, show_trades=False), use_container_width=True)
        
        st.markdown("---")
        st.subheader("😱 PTT 恐慌時刻熱門話題 (Top 20)")
        
        # 🔥 修改：改為提取 20 筆
        hot_news = extract_hot_news_titles(df, 20) 
        
        if hot_news:
            # 使用兩欄排列，左新右舊，或者左右交錯
            c1, c2 = st.columns(2)
            for i, news in enumerate(hot_news):
                with (c1 if i % 2 == 0 else c2):
                    ratio_val = float(news['ratio'])
                    # 顏色區分：大於 3.0 用深紅，其餘用橘色
                    badge_color = "#D62728" if ratio_val > 3.0 else "#FF7F0E"
                    
                    st.markdown(
                        f"<div style='margin-bottom:12px; padding:12px; border-radius:6px; background-color:#f8f9fa; border-left: 5px solid {badge_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1)'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px'>"
                        f"<small style='color:#666; font-weight:bold'>{news['date']}</small>"
                        f"<span style='background-color:{badge_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75em; font-weight:bold'>Ratio: {news['ratio']}</span>"
                        f"</div>"
                        f"<div style='font-size:1.05em; font-weight:500; color:#333'>{news['title']}</div>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
        else:
            st.info("此區間無足夠情緒數據。")
    else:
        st.warning("⚠️ 此時間範圍內無數據，請調整側邊欄日期設定。")

# ==================== Tab 2: 策略模式 (進階) ====================
with tab2:
    st.markdown("### 🧪 策略回測實驗室")
    st.caption("模擬若依照「PTT 恐慌情緒買進」策略的歷史績效。")
    
    if not df.empty:
        with st.expander("⚙️ 調整策略參數", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                hold_days = st.slider("最大持有天數", 5, 60, 20)
                cost_rate = st.slider("交易成本 (%)", 0.0, 1.0, 0.2)
            with c2:
                stop_loss = st.slider("停損比例 (%)", 2.0, 20.0, 7.0)
            with c3:
                take_profit = st.slider("停利比例 (%)", 5.0, 50.0, 15.0)

        trades_df = robust_strategy_backtest(df, hold_days, cost_rate, stop_loss, take_profit)
        
        if not trades_df.empty:
            returns = trades_df['淨報酬']
            win_rate = (returns > 0).mean() * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("交易次數", len(trades_df))
            m2.metric("總報酬", f"{returns.sum():.2f}%", delta_color="normal")
            m3.metric("勝率", f"{win_rate:.1f}%")
            m4.metric("平均報酬", f"{returns.mean():.2f}%", 
                     delta=f"{returns.mean():.2f}%", delta_color="off")
            
            st.plotly_chart(create_main_chart(df, trades_df, show_trades=True), use_container_width=True)
            
            st.subheader("📝 交易明細")
            show_df = trades_df[['進場日期', '主要策略', '進場價格', '出場價格', '出場原因', '淨報酬', '持有天數']].copy()
            show_df['進場日期'] = show_df['進場日期'].dt.date
            show_df['進場價格'] = show_df['進場價格'].astype(int)
            show_df['出場價格'] = show_df['出場價格'].astype(int)
            show_df['淨報酬'] = show_df['淨報酬'].apply(lambda x: f"{x:.2f}%")
            
            def color_ret(val):
                color = '#ffcccc' if '-' in val else '#ccffcc'
                return f'background-color: {color}'
                
            st.dataframe(show_df.style.applymap(color_ret, subset=['淨報酬']), use_container_width=True)
        else:
            st.warning("⚠️ 在此參數與時間區間下，無觸發任何交易信號。")
    else:
        st.warning("⚠️ 請先選擇有效的時間範圍。")
