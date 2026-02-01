#!/usr/bin/env python3
"""
Emotion Cycle ETL v12 - 最終版
改進重點：
1. 排除法白名單：避開公告、閒聊
2. 保留更多有價值的情緒討論  
3. 詳細的過濾統計報告
"""

import pandas as pd
import yfinance as yf
import glob
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def run_etl():
    print("🚀 Starting Emotion Cycle ETL v12 - Final Edition")
    print("🎯 New: Smart whitelist + News filtering")
    print("=" * 60)
    
    # 1. 載入 PTT 數據
    print("📂 Loading PTT data...")
    df_ptt = load_ptt_data()
    if df_ptt is None:
        return False
    
    # 2. 聚合日度數據
    print("📊 Aggregating daily data...")
    daily_df = aggregate_daily_data(df_ptt)
    
    # 3. 計算情緒指標
    print("🧠 Calculating emotion metrics...")
    daily_df = calculate_emotion_metrics(daily_df)
    
    # 4. 下載市場數據
    print("📈 Fetching market data...")
    daily_df = add_market_data(daily_df)
    
    # 5. 週期分析（實時版本）
    print("🌊 Analyzing emotion cycles (real-time)...")
    daily_df = analyze_emotion_cycles_realtime(daily_df)
    
    # 6. 生成交易信號
    print("🎯 Generating trading signals...")
    daily_df = generate_trading_signals(daily_df)
    
    # 7. 添加熱門標題（改良版）
    print("📝 Adding hot news titles...")
    daily_df = add_enhanced_titles(daily_df, df_ptt)
    
    # 8. 保存結果
    output_file = 'emotion_cycle_ready_v12_final.csv'
    daily_df.to_csv(output_file, index=False)
    print(f"✅ Data saved to: {output_file}")
    
    # 9. 生成詳細報告
    print_comprehensive_report(daily_df, df_ptt)
    
    return True

def load_ptt_data():
    """載入並智慧過濾 PTT 數據"""
    csv_files = glob.glob('ptt_*.csv')
    if not csv_files:
        print("❌ No PTT data found (ptt_*.csv)")
        print("   Please run PTT crawler first")
        return None
    
    # 合併所有 CSV 檔案
    print(f"📁 Found {len(csv_files)} CSV files")
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            df_list.append(df)
            print(f"   ✓ {file}: {len(df)} records")
        except Exception as e:
            print(f"   ✗ {file}: Error - {e}")
    
    if not df_list:
        return None
        
    df_ptt = pd.concat(df_list, ignore_index=True)
    
    # 基本清理
    df_ptt['title'] = df_ptt['title'].astype(str)
    df_ptt['date'] = pd.to_datetime(df_ptt['date'], errors='coerce')
    df_ptt['energy'] = pd.to_numeric(df_ptt['energy'], errors='coerce')
    
    original_count = len(df_ptt)
    print(f"\n📊 Raw PTT records: {original_count:,}")
    
    # === 智慧白名單過濾系統 ===
    print("\n🧹 Smart Whitelist Filtering...")
    
    # 第一步：排除明確的無關內容
    exclude_keywords = [
        '公告', '閒聊', '問卦', 'Re:', 
        '[食物]', '[感情]', '[笑話]', '[八卦]',
        '徵友', '交友', '活動', '版規'
    ]
    
    for keyword in exclude_keywords:
        before_count = len(df_ptt)
        df_ptt = df_ptt[~df_ptt['title'].str.contains(keyword, case=False, na=False)]
        excluded = before_count - len(df_ptt)
        if excluded > 0:
            print(f"   🚫 Excluded '{keyword}': {excluded:,} posts")
    
    after_exclusion = len(df_ptt)
    
    # 第二步：保留有價值的內容（擴大範圍）
    valuable_keywords = [
        # 正式標籤
        '新聞', '情報', '標的', '請益', '心得', '討論',
        # 強情緒詞彙
        '跌', '漲', '崩', '噴', '飆', '慘', '爽', '完蛋', '恐慌', 
        '血流成河', '套牢', '停損', '逃命', '起飛', '發財', '賺翻',
        # 技術分析
        '支撐', '壓力', '突破', '反彈', '回檔', '轉強', '轉弱',
        # 市場動態  
        '融資', '融券', '法人', '外資', '投信', '主力',
        # 重要事件
        '財報', '除息', '除權', '減資', '增資', '合併',
        # 個股相關
        'ETF', '台積電', '鴻海', '聯發科', '台塑', '中鋼'
    ]
    
    # 保留包含有價值關鍵字的文章
    valuable_mask = df_ptt['title'].str.contains('|'.join(valuable_keywords), case=False, na=False)
    df_ptt_final = df_ptt[valuable_mask]
    
    final_count = len(df_ptt_final)
    
    # 過濾統計報告
    print(f"\n📈 Filtering Results:")
    print(f"   📥 Original posts: {original_count:,}")
    print(f"   🚫 After exclusion: {after_exclusion:,} (-{original_count-after_exclusion:,}, -{(original_count-after_exclusion)/original_count*100:.1f}%)")
    print(f"   ✅ Final valuable posts: {final_count:,} (-{original_count-final_count:,}, -{(original_count-final_count)/original_count*100:.1f}%)")
    print(f"   📊 Retention rate: {final_count/original_count*100:.1f}%")
    
    # 移除無效數據
    df_ptt_final = df_ptt_final.dropna(subset=['date', 'energy'])
    df_ptt_final['date'] = df_ptt_final['date'].dt.normalize()
    
    print(f"   🧹 After cleanup: {len(df_ptt_final):,} records")
    
    return df_ptt_final

def aggregate_daily_data(df_ptt):
    """聚合為日度數據"""
    df_ptt['energy_abs'] = df_ptt['energy'].abs()
    
    # 日度聚合 - 加入更多統計
    daily_agg = df_ptt.groupby('date').agg({
        'energy_abs': ['sum', 'mean', 'std'],
        'energy': ['count', 'mean'],
        'title': 'count'
    }).reset_index()
    
    # 扁平化欄位名稱
    daily_agg.columns = ['date', 'energy_abs', 'energy_mean', 'energy_std', 'post_count', 'avg_energy', 'title_count']
    
    # 創建完整的日期範圍
    full_range = pd.date_range(
        start=daily_agg['date'].min(),
        end=daily_agg['date'].max(),
        freq='D'
    )
    
    full_df = pd.DataFrame({'date': full_range})
    daily_df = full_df.merge(daily_agg, on='date', how='left')
    
    # 填充缺失值
    daily_df = daily_df.fillna({
        'energy_abs': 0,
        'energy_mean': 0, 
        'energy_std': 0,
        'post_count': 0,
        'avg_energy': 0,
        'title_count': 0
    })
    
    print(f"✅ Aggregated to {len(daily_df)} daily records")
    return daily_df

def calculate_emotion_metrics(df):
    """計算情緒相關指標"""
    
    # 基礎移動平均
    df['MA_7'] = df['energy_abs'].rolling(7, min_periods=1).mean()
    df['MA_30'] = df['energy_abs'].rolling(30, min_periods=1).mean()
    df['MA_60'] = df['energy_abs'].rolling(60, min_periods=1).mean()
    
    # 穩定基準線（防止早期數據分母過小）
    MIN_BASELINE = 50
    df['stable_MA'] = np.maximum(df['MA_30'], MIN_BASELINE)
    
    # 情緒強度比率
    df['ratio'] = df['energy_abs'] / df['stable_MA']
    
    # 情緒動量和加速度
    df['emotion_momentum'] = df['ratio'].diff()
    df['emotion_acceleration'] = df['emotion_momentum'].diff()
    
    # 情緒波動率
    df['emotion_volatility'] = df['ratio'].rolling(20, min_periods=5).std()
    
    # 情緒RSI
    df['emotion_rsi'] = calculate_rsi(df['ratio'], 14)
    
    # 情緒偏離度
    long_term_avg = df['ratio'].rolling(252, min_periods=50).mean()
    df['emotion_deviation'] = (df['ratio'] - long_term_avg) / long_term_avg
    df['emotion_deviation'] = df['emotion_deviation'].fillna(0)
    
    # 動態閾值（核心改進）
    df['panic_threshold'] = df['ratio'].rolling(252, min_periods=50).quantile(0.9)
    df['calm_threshold'] = df['ratio'].rolling(252, min_periods=50).quantile(0.7)
    df['extreme_threshold'] = df['ratio'].rolling(252, min_periods=50).quantile(0.95)
    
    # 填充初期閾值
    df['panic_threshold'] = df['panic_threshold'].fillna(2.5)
    df['calm_threshold'] = df['calm_threshold'].fillna(1.3)
    df['extreme_threshold'] = df['extreme_threshold'].fillna(3.5)
    
    return df

def calculate_rsi(series, period=14):
    """計算RSI指標"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def add_market_data(df):
    """添加市場價格數據"""
    try:
        # 下載台股指數
        start_date = df['date'].min() - timedelta(days=5)
        end_date = df['date'].max() + timedelta(days=5)
        
        print(f"   📅 Downloading TWII from {start_date.date()} to {end_date.date()}")
        market_df = yf.download('^TWII', start=start_date, end=end_date, progress=False)
        
        # 處理多層 column index
        if isinstance(market_df.columns, pd.MultiIndex):
            market_df.columns = [col[0] for col in market_df.columns.values]
            
        # 重設索引並標準化
        market_df = market_df.reset_index()
        market_df['Date'] = pd.to_datetime(market_df['Date']).dt.normalize()
        
        # 選取需要的欄位並重命名
        market_clean = market_df[['Date', 'Open', 'Close', 'Volume']].copy()
        market_clean.columns = ['date', 'twii_open', 'twii', 'twii_volume']
        
        # 合併到主數據框
        df = df.merge(market_clean, on='date', how='left')
        
        # 前向填充（避免未來數據洩露）
        df['twii'] = df['twii'].ffill()
        df['twii_open'] = df['twii_open'].ffill()
        df['twii_volume'] = df['twii_volume'].ffill()
        
        # 計算報酬率
        df['twii_ret'] = df['twii'].pct_change() * 100
        
        # 計算技術指標
        df['MA_200'] = df['twii'].rolling(200, min_periods=1).mean()
        
        valid_price_count = df['twii'].notna().sum()
        print(f"   ✅ Added market data: {valid_price_count}/{len(df)} valid price points")
        
    except Exception as e:
        print(f"   ⚠️ Market data download failed: {e}")
        # 創建假數據以便測試
        df['twii'] = 15000
        df['twii_open'] = 15000
        df['twii_volume'] = 1000000
        df['twii_ret'] = 0
        df['MA_200'] = 15000
    
    return df

def analyze_emotion_cycles_realtime(df):
    """實時週期檢測 - 消除後照鏡偏差"""
    
    df['cycle_phase'] = '正常期'
    df['emotion_intensity'] = '低'
    df['phase_confidence'] = 0.5
    
    for i in range(5, len(df)):  # 需要至少5天歷史
        current_ratio = df.iloc[i]['ratio']
        
        # 只看過去的趨勢（關鍵：不能看未來）
        recent_ratios = df.iloc[i-5:i]['ratio']  # 過去5天，不包含當天
        recent_trend = recent_ratios.diff().tail(3).mean()  # 近3天趨勢
        
        # 動態閾值
        panic_threshold = df.iloc[i]['panic_threshold']
        calm_threshold = df.iloc[i]['calm_threshold'] 
        extreme_threshold = df.iloc[i]['extreme_threshold']
        
        # 實時週期判斷
        confidence = 0.5
        
        if current_ratio > extreme_threshold:
            # 極端恐慌
            if recent_trend < -0.15:  # 強烈下降趨勢
                phase = '疲乏期'
                intensity = '極高轉中'
                confidence = 0.8
            else:
                phase = '極度恐慌'
                intensity = '極高'
                confidence = 0.9
                
        elif current_ratio > panic_threshold:
            # 恐慌區間
            if recent_trend < -0.1:  # 恐慌強度在下降
                phase = '疲乏期' 
                intensity = '高轉中'
                confidence = 0.7
            else:
                phase = '恐慌期'
                intensity = '高'
                confidence = 0.8
                
        elif current_ratio > calm_threshold:
            # 警戒區間
            phase = '警戒期'
            intensity = '中'
            confidence = 0.6
            
        else:
            # 冷靜區間
            if recent_trend > 0.05:  # 從低點回升
                phase = '復甦期'
                intensity = '低轉中' 
                confidence = 0.7
            else:
                phase = '正常期'
                intensity = '低'
                confidence = 0.5
        
        # 更新數據
        df.iloc[i, df.columns.get_loc('cycle_phase')] = phase
        df.iloc[i, df.columns.get_loc('emotion_intensity')] = intensity
        df.iloc[i, df.columns.get_loc('phase_confidence')] = confidence
    
    print(f"   ✅ Real-time cycle analysis completed")
    return df

def generate_trading_signals(df):
    """生成交易信號"""
    
    df['buy_signal_cycle'] = False
    df['sell_signal_cycle'] = False
    df['signal_confidence'] = 0.0
    df['signal_reason'] = ''
    
    for i in range(10, len(df)):  # 需要足夠歷史數據
        
        current_phase = df.iloc[i]['cycle_phase']
        current_ratio = df.iloc[i]['ratio']
        
        # 疲乏期買入信號
        if current_phase == '疲乏期':
            confirmations = []
            
            # 1. 趨勢確認：連續下降
            if i >= 2:
                trend_confirm = (df.iloc[i-2]['ratio'] > df.iloc[i-1]['ratio'] > df.iloc[i]['ratio'])
                if trend_confirm:
                    confirmations.append('trend_declining')
            
            # 2. 相對位置：不在極高位
            recent_max = df.iloc[i-5:i]['ratio'].max()
            if current_ratio < recent_max * 0.8:
                confirmations.append('relative_position')
            
            # 3. 動態閾值確認
            if current_ratio < df.iloc[i]['panic_threshold'] * 0.9:
                confirmations.append('threshold_check')
            
            # 4. RSI 超賣確認
            if pd.notna(df.iloc[i]['emotion_rsi']) and df.iloc[i]['emotion_rsi'] < 35:
                confirmations.append('rsi_oversold')
            
            # 信號生成
            confidence = len(confirmations) / 4.0
            
            if len(confirmations) >= 2:
                df.iloc[i, df.columns.get_loc('buy_signal_cycle')] = True
                df.iloc[i, df.columns.get_loc('signal_confidence')] = confidence
                df.iloc[i, df.columns.get_loc('signal_reason')] = '+'.join(confirmations)
        
        # 復甦期賣出信號
        elif current_phase == '復甦期':
            if current_ratio < df.iloc[i]['calm_threshold'] * 1.1:
                df.iloc[i, df.columns.get_loc('sell_signal_cycle')] = True
                df.iloc[i, df.columns.get_loc('signal_confidence')] = 0.7
                df.iloc[i, df.columns.get_loc('signal_reason')] = 'recovery_exit'
    
    signal_count = df['buy_signal_cycle'].sum()
    print(f"   ✅ Generated {signal_count} buy signals")
    
    return df

def add_enhanced_titles(df, df_ptt):
    """添加增強版熱門標題"""
    
    # 按日期和能量排序
    df_ptt_sorted = df_ptt.sort_values(['date', 'energy_abs'], ascending=[True, False])
    
    # 為每日提取新聞標題
    title_map = {}
    news_map = {}
    
    for date, group in df_ptt_sorted.groupby('date'):
        all_titles = []
        news_titles = []
        
        for _, row in group.head(10).iterrows():  # 提取前10個
            energy = int(row['energy']) if pd.notna(row['energy']) else 0
            title = str(row['title'])
            
            # 清理標題
            clean_title = title.replace('\n', ' ').replace('\r', ' ').strip()
            if len(clean_title) > 80:
                clean_title = clean_title[:80] + "..."
            
            formatted_title = f"🔥[{energy}] {clean_title}"
            all_titles.append(formatted_title)
            
            # 單獨提取新聞標題
            if '[新聞]' in title:
                news_titles.append(formatted_title)
        
        title_map[date] = "<br>".join(all_titles)
        news_map[date] = "<br>".join(news_titles[:5])  # 只要前5條新聞
    
    # 映射到主數據框
    df['top_titles'] = df['date'].map(title_map).fillna("No data")
    df['news_titles'] = df['date'].map(news_map).fillna("No news")
    
    news_days = (df['news_titles'] != "No news").sum()
    print(f"   ✅ Added titles for {len(title_map)} days, {news_days} days with news")
    
    return df

def print_comprehensive_report(df, df_ptt):
    """生成綜合分析報告"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE ETL REPORT - EMOTION CYCLE HUNTER v12")
    print("="*80)
    
    # 基本統計
    print(f"\n📅 Data Overview:")
    print(f"   • Analysis period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   • Total trading days: {len(df):,}")
    print(f"   • Average daily emotion: {df['ratio'].mean():.2f}")
    print(f"   • Peak emotion level: {df['ratio'].max():.2f}")
    print(f"   • 90th percentile: {df['ratio'].quantile(0.9):.2f}")
    
    # 週期統計
    print(f"\n🌊 Emotion Cycle Analysis:")
    if 'cycle_phase' in df.columns:
        phase_counts = df['cycle_phase'].value_counts()
        total_days = len(df)
        
        for phase, count in phase_counts.items():
            pct = count / total_days * 100
            print(f"   • {phase}: {count:,} days ({pct:.1f}%)")
            
        # 疲乏期統計
        fatigue_days = phase_counts.get('疲乏期', 0)
        if fatigue_days > 0:
            fatigue_ratio = df[df['cycle_phase'] == '疲乏期']['ratio'].mean()
            print(f"\n   🎯 Fatigue Period Insights:")
            print(f"     - Total fatigue days: {fatigue_days}")
            print(f"     - Avg emotion in fatigue: {fatigue_ratio:.2f}")
            print(f"     - Fatigue frequency: every {total_days/fatigue_days:.1f} days")
    
    # 信號統計
    print(f"\n🎯 Trading Signal Analysis:")
    if 'buy_signal_cycle' in df.columns:
        buy_signals = df['buy_signal_cycle'].sum()
        
        if buy_signals > 0:
            avg_confidence = df[df['buy_signal_cycle']]['signal_confidence'].mean()
            signal_freq = len(df) / buy_signals
            
            print(f"   • Total buy signals: {buy_signals}")
            print(f"   • Average confidence: {avg_confidence:.2f}")
            print(f"   • Signal frequency: every {signal_freq:.1f} days")
            
            # 信號原因分析
            reasons = df[df['buy_signal_cycle']]['signal_reason'].value_counts()
            if not reasons.empty:
                print(f"   • Top signal reasons:")
                for reason, count in reasons.head(3).items():
                    print(f"     - {reason}: {count} times")
        else:
            print(f"   • No buy signals generated in this period")
    
    # 新聞統計
    print(f"\n📰 News Content Analysis:")
    if 'news_titles' in df.columns:
        news_days = (df['news_titles'] != "No news").sum()
        print(f"   • Days with news content: {news_days:,}")
        print(f"   • News coverage rate: {news_days/len(df)*100:.1f}%")
        
        # 分析新聞關鍵字
        all_news = df[df['news_titles'] != "No news"]['news_titles'].str.cat(sep=' ')
        keywords = ['跌', '漲', '崩', '噴', '恐慌', '利多', '突破']
        
        print(f"   • News keyword frequency:")
        for keyword in keywords:
            count = all_news.count(keyword)
            if count > 0:
                print(f"     - '{keyword}': {count} mentions")
    
    # 動態閾值分析
    print(f"\n📊 Dynamic Threshold Analysis:")
    if 'panic_threshold' in df.columns:
        avg_panic = df['panic_threshold'].mean()
        avg_calm = df['calm_threshold'].mean()
        threshold_volatility = df['panic_threshold'].std()
        
        print(f"   • Average panic threshold: {avg_panic:.2f}")
        print(f"   • Average calm threshold: {avg_calm:.2f}")
        print(f"   • Threshold volatility: {threshold_volatility:.2f}")
        print(f"   • Adaptive range: {avg_panic - avg_calm:.2f}")
    
    # 數據品質報告
    print(f"\n✅ Data Quality Report:")
    print(f"   • Missing emotion data: {df['ratio'].isna().sum()} days")
    print(f"   • Missing price data: {df['twii'].isna().sum()} days")
    print(f"   • Data completeness: {(1 - df[['ratio', 'twii']].isna().any(axis=1).mean())*100:.1f}%")
    
    # 建議
    print(f"\n💡 Strategy Recommendations:")
    
    if 'buy_signal_cycle' in df.columns and df['buy_signal_cycle'].sum() > 0:
        high_confidence_signals = df[(df['buy_signal_cycle']) & (df['signal_confidence'] > 0.7)].shape[0]
        total_signals = df['buy_signal_cycle'].sum()
        
        print(f"   • Focus on high-confidence signals: {high_confidence_signals}/{total_signals} ({high_confidence_signals/total_signals*100:.1f}%)")
        print(f"   • Consider dynamic position sizing based on signal confidence")
        print(f"   • Monitor news sentiment for signal validation")
    else:
        print(f"   • Consider lowering signal thresholds or extending analysis period")
        print(f"   • Review emotion cycle patterns for optimization opportunities")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run: streamlit run emotion_cycle_dashboard_v12_improved.py")
    print(f"   2. Analyze signal quality in different market conditions")  
    print(f"   3. Backtest with proper position sizing")
    print(f"   4. Monitor real-time performance")
    
    print("="*80)

if __name__ == "__main__":
    success = run_etl()
    if success:
        print("\n✅ ETL completed successfully!")
        print("🎯 Ready for dashboard analysis")
    else:
        print("\n❌ ETL failed. Please check error messages above.")