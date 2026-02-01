import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import re
from datetime import datetime
import sys

class PTTReverseCrawler:
    def __init__(self):
        self.base_url = "https://www.ptt.cc/bbs/Stock/index.html"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'cookie': 'over18=1' # 這是必備的
        }
        self.data_buffer = []
        
        # 🔥 修改 1: 建立強大的連線 Session (解決 ConnectionResetError)
        self.session = requests.Session()
        # 設定重試策略：遇到 500/502/503/504 或斷線時，最多重試 5 次，每次間隔時間指數增加
        retries = Retry(
            total=5, 
            backoff_factor=1, 
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'POST'])
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.headers.update(self.headers)
        
        self.current_year = datetime.now().year
        # 🔥 修改 2: 設定目標年份為 2019
        self.target_min_year = 2019
        
        # 🔥 修改 3: 增加 GitHub Actions 專用的快速模式
        if len(sys.argv) > 1 and sys.argv[1] == 'current_only':
            self.target_min_year = self.current_year
            print("⚡ 啟用快速模式：僅更新當前年份資料")
            
        print(f"🤖 爬蟲初始化完成。當前年份：{self.current_year}，目標年份：{self.target_min_year}")

    def get_last_page_number(self):
        """抓取 Stock 版最新的頁碼 (包含錯誤處理)"""
        try:
            # 使用 session.get 而不是 requests.get
            res = self.session.get(self.base_url, timeout=15)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            paging_div = soup.find('div', 'btn-group-paging')
            if not paging_div:
                print("❌ 無法找到分頁按鈕，可能被 IP Ban 或結構改變")
                return None
            
            links = paging_div.find_all('a')
            prev_link = links[1]['href']
            
            match = re.search(r'index(\d+)', prev_link)
            if match:
                return int(match.group(1)) + 1
            return None
        except Exception as e:
            print(f"❌ 初始化失敗 (無法取得最新頁碼): {e}")
            return None

    def parse_energy(self, push_str):
        if not push_str: return 0
        if push_str == '爆': return 100
        if 'X' in push_str: return -10
        try: return int(push_str)
        except: return 0

    def save_year_data(self, year):
        """將緩衝區資料寫入 CSV (包含去重功能)"""
        if not self.data_buffer: return
        
        filename = f"ptt_{year}.csv"
        new_df = pd.DataFrame(self.data_buffer)
        
        # 讀取舊檔並合併去重
        if os.path.isfile(filename):
            try:
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # 依據日期和標題去重，保留最新的
                combined_df = combined_df.drop_duplicates(subset=['date', 'title'], keep='last')
            except:
                combined_df = new_df
        else:
            combined_df = new_df
        
        # 依日期排序
        combined_df = combined_df.sort_values('date')
        combined_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"💾 [存檔成功] {filename} | 目前總筆數 {len(combined_df)}")
        self.data_buffer = [] 

    def run(self):
        start_page = self.get_last_page_number()
        if not start_page: return

        print(f"🚀 啟動時光機：從第 {start_page} 頁 ({self.current_year}) 開始...")
        
        prev_month = None 
        
        # 這裡不需要改，只要確保 session 穩定即可
        for page in range(start_page, 1, -1):
            url = f"https://www.ptt.cc/bbs/Stock/index{page}.html"
            try:
                res = self.session.get(url, timeout=10)
                if res.status_code != 200:
                    print(f"⚠️ Page {page} status: {res.status_code}")
                    continue
            except Exception as e:
                print(f"⚠️ Page {page} 連線錯誤 (已重試): {e}")
                time.sleep(5)
                continue

            soup = BeautifulSoup(res.text, 'html.parser')
            divs = soup.find_all('div', 'r-ent')
            
            for d in reversed(divs):
                try:
                    title_div = d.find('div', 'title')
                    if not title_div or not title_div.find('a'): continue
                    
                    title = title_div.find('a').text.strip()
                    date_str = d.find('div', 'date').text.strip()
                    push_count = self.parse_energy(d.find('div', 'nrec').text)
                    
                    date_parts = date_str.split('/')
                    if len(date_parts) < 2: continue
                    month = int(date_parts[0])
                    
                    if prev_month is None: prev_month = month
                    
                    # 偵測跨年
                    if prev_month == 1 and month == 12:
                        print(f"📅 偵測到跨年！從 {self.current_year} -> {self.current_year - 1}")
                        self.save_year_data(self.current_year)
                        self.current_year -= 1
                        
                        if self.current_year < self.target_min_year:
                            print("🎉 已完成目標年份抓取。")
                            return

                    prev_month = month
                    full_date = f"{self.current_year}/{date_str.strip()}"
                    
                    self.data_buffer.append({
                        'date': full_date,
                        'title': title,
                        'energy': push_count
                    })

                except: continue
            
            # 進度顯示與存檔
            if page % 20 == 0:
                print(f"處理中: Page {page} | 年份: {self.current_year}")
                self.save_year_data(self.current_year)
                time.sleep(random.uniform(0.5, 1.0)) # 線上執行稍微加快
            else:
                # 隨機延遲，避免被鎖
                time.sleep(random.uniform(0.1, 0.2))

        self.save_year_data(self.current_year)
        print("✅ 爬蟲執行完畢。")

if __name__ == "__main__":
    crawler = PTTReverseCrawler()
    crawler.run()