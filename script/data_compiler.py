#!/usr/bin/env python3
import argparse
import logging
import struct
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

WIRE_ORDER_FORMAT = '<qiiiq36x'
WIRE_ORDER_SIZE = 64

PRICE_TICK_MULTIPLIER = 1
QTY_TICK_MULTIPLIER = 1000
CSV_CHUNK_SIZE = 500_000


def download_binance_data(symbol: str, year: str, month: str, dest_dir: Path) -> Path:
    filename = f"{symbol}-trades-{year}-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/trades/{symbol}/{filename}"

    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = dest_dir / filename.replace('.zip', '.csv')
    if raw_csv_path.exists():
        logging.info(f"CSV file already exists: {raw_csv_path}")
        return raw_csv_path

    logging.info(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 404:
        logging.error(f"Data not found on Binance for {symbol} {year}-{month}.")
        sys.exit(1)

    response.raise_for_status()
    logging.info("Extracting ZIP archive...")
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        archive.extractall(dest_dir)

    return raw_csv_path


def process_and_pack_chunk(df_chunk: pd.DataFrame, file_handle) -> int:
    n = len(df_chunk)

    timestamp_ns = (df_chunk['time'].to_numpy(dtype=np.int64)) * 1_000_000
    side = np.where(df_chunk['is_buyer_maker'].to_numpy(dtype=bool), 0, 1).astype(np.int32)
    price_ticks = (df_chunk['price'].to_numpy(dtype=np.float64) * PRICE_TICK_MULTIPLIER).astype(np.int32)
    qty_ticks = (df_chunk['qty'].to_numpy(dtype=np.float64) * QTY_TICK_MULTIPLIER).astype(np.int32)
    user_id = df_chunk['id'].to_numpy(dtype=np.int64)

    buf = np.zeros(n, dtype=np.dtype([
        ('timestamp_ns', '<i8'),
        ('side', '<i4'),
        ('price_ticks', '<i4'),
        ('qty_ticks', '<i4'),
        ('user_id', '<i8'),
        ('_pad', 'V36'),
    ]))

    buf['timestamp_ns'] = timestamp_ns
    buf['side'] = side
    buf['price_ticks'] = price_ticks
    buf['qty_ticks'] = qty_ticks
    buf['user_id'] = user_id

    buf.tofile(file_handle)
    return n


def compile_to_binary(csv_path: Path, bin_path: Path) -> None:
    logging.info(f"Reading CSV dataset: {csv_path}")

    csv_columns = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    total_processed = 0
    logging.info("Starting chunked compilation...")

    with open(bin_path, 'wb') as f_out:
        chunk_iterator = pd.read_csv(csv_path, names=csv_columns, chunksize=CSV_CHUNK_SIZE)
        for chunk_idx, df_chunk in enumerate(chunk_iterator):
            processed = process_and_pack_chunk(df_chunk, f_out)
            total_processed += processed
            logging.info(f"Processed chunk {chunk_idx + 1} | Total orders serialized: {total_processed:,}")

    size_mb = bin_path.stat().st_size / (1024 * 1024)
    logging.info(f"Compilation successful: {bin_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Historical Data Compiler")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--year', type=str, default='2025', help='Target year (e.g., 2025)')
    parser.add_argument('--month', type=str, default='01', help='Target month format MM (e.g., 01)')
    parser.add_argument('--out', type=str, default=None, help='Custom output binary path')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent / 'data'
    raw_dir = base_dir / 'raw'

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = base_dir / f"{args.symbol}_{args.year}_{args.month}.bin"

    try:
        csv_path = download_binance_data(args.symbol, args.year, args.month, raw_dir)
        compile_to_binary(csv_path, out_path)
    except Exception as e:
        logging.error(f"Fatal compilation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
