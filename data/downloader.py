import os
import time
import json
import urllib.request
import pandas as pd
from datetime import datetime


class BinanceDownloader:

    BASE_URL_SPOT = 'https://api.binance.com/api/v3/klines'
    BASE_URL_FUTURES = 'https://fapi.binance.com/fapi/v1/klines'

    def __init__(self, data_dir='data_integrated'):
        self.data_dir = data_dir
        self.spot_dir = os.path.join(data_dir, 'spot')
        self.futures_dir = os.path.join(data_dir, 'futures')
        for d in [data_dir, self.spot_dir, self.futures_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def fetch_data(self, symbol, start_ts, end_ts, data_type='futures'):
        url_base = self.BASE_URL_FUTURES if data_type == 'futures' else self.BASE_URL_SPOT
        all_data = []
        current_start = start_ts

        while current_start < end_ts:
            params = f"?symbol={symbol}&interval=1m&startTime={current_start}&endTime={end_ts}&limit=1000"
            url = url_base + params
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.status != 200:
                        time.sleep(1)
                        continue
                    data = json.loads(response.read().decode())
            except Exception:
                time.sleep(2)
                continue

            if not data:
                break
            all_data.extend(data)
            current_start = data[-1][0] + 60000
            time.sleep(0.05)

        return all_data

    def process_and_save(self, raw_data, filepath):
        if not raw_data:
            return
        df = pd.DataFrame(raw_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore',
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        df[['open', 'high', 'low', 'close', 'volume']].to_csv(filepath)

    def run(self, symbols, start_date='2021-01-01', end_date=None):
        end_dt = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        for symbol in symbols:
            for dtype, ddir in [('spot', self.spot_dir), ('futures', self.futures_dir)]:
                path = os.path.join(ddir, f"{symbol}.csv")
                if not os.path.exists(path):
                    raw = self.fetch_data(symbol, start_ts, end_ts, dtype)
                    self.process_and_save(raw, path)


def download_binance_data(config):
    symbols = config.get('symbols', [])
    if not symbols:
        raise ValueError("No symbols specified in config")
    downloader = BinanceDownloader(data_dir=config.get('data_dir', 'data_integrated'))
    downloader.run(
        symbols=symbols,
        start_date=config.get('start_date', '2021-01-01'),
        end_date=config.get('end_date'),
    )