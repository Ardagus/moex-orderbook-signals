"""Офлайн-тесты сборщика: на синтетических сообщениях, без сети и токена."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import collect  # noqa: E402
from collect import common_pb2, md  # noqa: E402


def make_book():
    ob = md.OrderBook(ticker="SBER", depth=10, is_consistent=True)
    ob.time.seconds, ob.time.nanos = 1_700_000_000, 5
    ob.bids.add(price=common_pb2.Quotation(units=250, nano=500_000_000), quantity=7)  # 250.5
    ob.asks.add(price=common_pb2.Quotation(units=250, nano=600_000_000), quantity=3)
    return ob


def test_book_row_converts_prices_and_pads_levels():
    row = collect.book_row(make_book(), recv_ns=1_700_000_001_000_000_000)
    assert row["bid_px_1"] == 250.5 and row["bid_qty_1"] == 7
    assert row["ask_px_1"] == 250.6 and row["ask_qty_1"] == 3
    assert pd.isna(row["bid_px_2"]) and row["bid_qty_2"] == 0  # уровня 2 нет -> NaN и 0
    assert row["exch_ts"] == 1_700_000_000_000_000_005


def test_trade_row_direction_and_price():
    tr = md.Trade(ticker="SBER", direction=md.TRADE_DIRECTION_BUY, quantity=4,
                  price=common_pb2.Quotation(units=250, nano=550_000_000))
    row = collect.trade_row(tr, recv_ns=1)
    assert row["trade_dir"] == "TRADE_DIRECTION_BUY" and row["trade_px"] == 250.55 and row["trade_qty"] == 4


def test_parts_merge_into_one_daily_file(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "RAW_DIR", tmp_path)
    monkeypatch.setattr(collect, "ROOT", tmp_path)  # merge_parts пишет относительные пути в лог
    w = collect.Writer()
    t0 = 1_700_000_000_000_000_000
    w.add(collect.book_row(make_book(), t0 + 2))
    w.maybe_flush(force=True)
    w.add(collect.book_row(make_book(), t0 + 1))  # вторая часть, но время раньше
    w.maybe_flush(force=True)
    collect.merge_parts()
    files = list((tmp_path / "SBER").glob("*.parquet"))
    assert len(files) == 1 and not list((tmp_path / "SBER").glob("*.parts"))
    df = pd.read_parquet(files[0])
    assert list(df["recv_ts"]) == [t0 + 1, t0 + 2]  # отсортировано по времени получения


def test_collect_reconnects_after_errors_and_saves_buffer(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "RAW_DIR", tmp_path)
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)  # не ждём паузу между попытками
    calls = []

    def fake_run_once(tickers, handle, deadline):
        calls.append(1)
        if len(calls) <= 2:
            raise ConnectionError("обрыв")  # два обрыва подряд
        handle(md.MarketDataResponse(orderbook=make_book()), 1_700_000_000_000_000_000)
        raise KeyboardInterrupt  # имитируем Ctrl+C после успешного переподключения

    monkeypatch.setattr(collect, "run_once", fake_run_once)
    collect.collect(["SBER"], minutes=1)
    assert len(calls) == 3  # после обрывов пробует снова
    df = pd.read_parquet(next((tmp_path / "SBER").glob("*.parquet")))
    assert len(df) == 1  # данные после переподключения сохранены
