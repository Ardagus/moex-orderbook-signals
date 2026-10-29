"""Сбор стакана (глубина 10) и ленты сделок MOEX через gRPC T-Invest API (read-only).

Запуск:
    python collect.py --check            # быстрая проверка подключения (20 с, файлы не пишет)
    python collect.py --minutes 15       # пробный сбор
    python collect.py                    # сбор до Ctrl+C

Переменные окружения:
    TINVEST_TOKEN        read-only токен (обязательно)
    TINVEST_CA_BUNDLE    путь к PEM с корнем "Russian Trusted Root CA" (сервер подписан им)
    TINVEST_ENDPOINT     по умолчанию invest-public-api.tinkoff.ru:443
"""
import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import grpc
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "tinvest_pb"))  # protoc генерирует импорты вида `import common_pb2`
import common_pb2  # noqa: E402
import marketdata_pb2 as md  # noqa: E402
import marketdata_pb2_grpc  # noqa: E402

TICKERS = ["SBER", "GAZP", "LKOH", "YDEX", "VTBR"]  # YDEX — бывший YNDX (после редомициляции)
CLASS_CODE = "TQBR"  # основной режим торгов акциями MOEX
DEPTH = 10
RAW_DIR = ROOT / "data" / "raw"
MSK = timezone(timedelta(hours=3))  # у Москвы нет перехода на летнее время
FLUSH_EVERY_S = 30
SILENCE_LIMIT_S = 30  # если столько секунд нет вообще никаких сообщений (даже ping) — рвём стрим
PING_DELAY_MS = 5000  # просим сервер слать ping часто, иначе по умолчанию он молчит до 120 с

log = logging.getLogger("collect")


def setup_logging():
    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_h = RotatingFileHandler(ROOT / "logs" / "collect.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    console_h = logging.StreamHandler()
    for h in (file_h, console_h):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def make_channel():
    endpoint = os.environ.get("TINVEST_ENDPOINT", "invest-public-api.tinkoff.ru:443")
    ca_path = os.environ.get("TINVEST_CA_BUNDLE")
    # доверяем корню только в этом канале, системное хранилище сертификатов не трогаем
    root_certs = Path(ca_path).read_bytes() if ca_path else None
    return grpc.secure_channel(endpoint, grpc.ssl_channel_credentials(root_certificates=root_certs))


def to_float(q):
    return q.units + q.nano / 1e9  # Quotation = целая часть + нано-доли


def ts_ns(t):
    return t.seconds * 1_000_000_000 + t.nanos


# ---------- превращение сообщений в строки ----------

def book_row(ob, recv_ns):
    row = {"kind": "book", "ticker": ob.ticker, "recv_ts": recv_ns, "exch_ts": ts_ns(ob.time),
           "is_consistent": ob.is_consistent}
    for i in range(DEPTH):  # недостающие уровни остаются NaN
        bid = ob.bids[i] if i < len(ob.bids) else None
        ask = ob.asks[i] if i < len(ob.asks) else None
        row[f"bid_px_{i + 1}"] = to_float(bid.price) if bid else float("nan")
        row[f"bid_qty_{i + 1}"] = bid.quantity if bid else 0
        row[f"ask_px_{i + 1}"] = to_float(ask.price) if ask else float("nan")
        row[f"ask_qty_{i + 1}"] = ask.quantity if ask else 0
    return row


def trade_row(tr, recv_ns):
    return {"kind": "trade", "ticker": tr.ticker, "recv_ts": recv_ns, "exch_ts": ts_ns(tr.time),
            "trade_px": to_float(tr.price), "trade_qty": tr.quantity,
            "trade_dir": md.TradeDirection.Name(tr.direction)}  # BUY/SELL — сторона агрессора


# ---------- запись на диск: части -> один parquet в день ----------

class Writer:
    def __init__(self):
        self.buf = {}  # ticker -> список строк
        self.last_flush = time.time()

    def add(self, row):
        self.buf.setdefault(row["ticker"], []).append(row)

    def maybe_flush(self, force=False):
        if not force and time.time() - self.last_flush < FLUSH_EVERY_S:
            return
        for ticker, rows in self.buf.items():
            if rows:
                self._write_part(ticker, rows)
        self.buf = {}
        self.last_flush = time.time()

    def _write_part(self, ticker, rows):
        # parquet нельзя дописывать, поэтому пишем маленькие части (при обрыве теряем максимум 30 с)
        day = datetime.fromtimestamp(rows[0]["recv_ts"] / 1e9, MSK).strftime("%Y-%m-%d")
        part_dir = RAW_DIR / ticker / f"{day}.parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(part_dir / f"{rows[0]['recv_ts']}.parquet", index=False)


def merge_parts():
    """Склеивает части в data/raw/{ticker}/{date}.parquet (вызывается при старте и выходе)."""
    for part_dir in RAW_DIR.glob("*/*.parts"):
        target = part_dir.with_suffix("").with_suffix(".parquet")  # 2026-10-29.parts -> 2026-10-29.parquet
        files = sorted(part_dir.glob("*.parquet"))
        frames = ([pd.read_parquet(target)] if target.exists() else []) + [pd.read_parquet(f) for f in files]
        pd.concat(frames, ignore_index=True).sort_values("recv_ts", kind="stable").to_parquet(target, index=False)
        for f in files:
            f.unlink()
        part_dir.rmdir()
        log.info("merged %s (%d rows)", target.relative_to(ROOT), sum(len(f) for f in frames))


# ---------- стрим ----------

def subscribe_requests(tickers, stop):
    """Генератор запросов: подписки, затем держим стрим открытым до stop."""
    ids = [f"{t}_{CLASS_CODE}" for t in tickers]
    yield md.MarketDataRequest(ping_settings=common_pb2.PingDelaySettings(ping_delay_ms=PING_DELAY_MS))
    yield md.MarketDataRequest(subscribe_order_book_request=md.SubscribeOrderBookRequest(
        subscription_action=md.SUBSCRIPTION_ACTION_SUBSCRIBE,
        instruments=[md.OrderBookInstrument(instrument_id=i, depth=DEPTH,
                                            order_book_type=md.ORDERBOOK_TYPE_EXCHANGE) for i in ids]))
    yield md.MarketDataRequest(subscribe_trades_request=md.SubscribeTradesRequest(
        subscription_action=md.SUBSCRIPTION_ACTION_SUBSCRIBE,
        instruments=[md.TradeInstrument(instrument_id=i) for i in ids],
        trade_source=md.TRADE_SOURCE_EXCHANGE))
    stop.wait()  # пока не остановят — стрим не закрываем


def log_subscriptions(resp):
    if resp.HasField("subscribe_order_book_response"):
        for s in resp.subscribe_order_book_response.order_book_subscriptions:
            log.info("orderbook sub %s_%s: %s", s.ticker, s.class_code, md.SubscriptionStatus.Name(s.subscription_status))
    else:
        for s in resp.subscribe_trades_response.trade_subscriptions:
            log.info("trades sub %s: %s", s.ticker, md.SubscriptionStatus.Name(s.subscription_status))


def run_once(tickers, handle, deadline):
    """Одно соединение со стримом. Возвращается при дедлайне, бросает исключение при обрыве."""
    token = os.environ["TINVEST_TOKEN"]
    channel = make_channel()
    stub = marketdata_pb2_grpc.MarketDataStreamServiceStub(channel)
    stop = threading.Event()
    last_msg = [time.time()]
    call = stub.MarketDataStream(subscribe_requests(tickers, stop), metadata=(("authorization", f"Bearer {token}"),))

    def watchdog():  # молчащий стрим gRPC сам не заметит, поэтому сторожим вручную
        while not stop.wait(5):
            if time.time() - last_msg[0] > SILENCE_LIMIT_S or (deadline and time.time() > deadline):
                call.cancel()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        for resp in call:
            last_msg[0] = time.time()
            handle(resp, time.time_ns())
    except grpc.RpcError as e:
        if deadline and time.time() > deadline:
            return  # нормальное завершение по таймеру
        raise ConnectionError(f"{e.code().name}: {e.details()}") from e
    finally:
        stop.set()
        channel.close()
    raise ConnectionError("стрим закрыт сервером")


def collect(tickers, minutes=None, write=True, on_msg=None):
    writer = Writer()
    counts = {}
    state = {"last_report": time.time()}

    def handle(resp, recv_ns):
        kind = resp.WhichOneof("payload")
        if on_msg:
            on_msg(kind, resp)
        if kind in ("subscribe_order_book_response", "subscribe_trades_response"):
            log_subscriptions(resp)
        elif kind == "orderbook":
            counts.setdefault(resp.orderbook.ticker, {"book": 0, "trade": 0})["book"] += 1
            if write:
                writer.add(book_row(resp.orderbook, recv_ns))
        elif kind == "trade":
            counts.setdefault(resp.trade.ticker, {"book": 0, "trade": 0})["trade"] += 1
            if write:
                writer.add(trade_row(resp.trade, recv_ns))
        elif kind is None and not state.get("warned"):
            # payload, которого нет в наших proto (контракты могли отстать от сервера) — смотрим сырые байты один раз
            log.info("unknown payload, raw bytes: %s", resp.SerializeToString().hex())
            state["warned"] = True
        if write:
            writer.maybe_flush()
        if time.time() - state["last_report"] > 60:
            log.info("counts: %s", counts)
            state["last_report"] = time.time()

    deadline = time.time() + minutes * 60 if minutes else None
    backoff = 1
    try:
        while not (deadline and time.time() > deadline):
            started = time.time()
            try:
                log.info("connecting, tickers=%s", tickers)
                run_once(tickers, handle, deadline)
            except Exception as e:  # любая ошибка -> лог и переподключение
                log.warning("stream error: %s", e)
                if write:
                    writer.maybe_flush(force=True)  # не теряем буфер при обрыве
                backoff = 1 if time.time() - started > 60 else min(backoff * 2, 30)  # долго жили -> сброс паузы
                log.info("reconnect in %ds", backoff)
                time.sleep(backoff)
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        if write:
            writer.maybe_flush(force=True)
            merge_parts()
    return counts


def check_connection():
    """Минимальная проверка: подписка на SBER, смотрим статусы подписок и первые сообщения."""
    seen = {}

    def on_msg(kind, resp):
        seen[kind] = seen.get(kind, 0) + 1
        if kind == "orderbook" and seen[kind] == 1:
            ob = resp.orderbook
            log.info("first book %s: depth=%d consistent=%s bid1=%s x%d ask1=%s x%d exch_time=%s",
                     ob.ticker, ob.depth, ob.is_consistent, to_float(ob.bids[0].price) if ob.bids else None,
                     ob.bids[0].quantity if ob.bids else 0, to_float(ob.asks[0].price) if ob.asks else None,
                     ob.asks[0].quantity if ob.asks else 0, ob.time.ToDatetime())

    collect(["SBER"], minutes=20 / 60, write=False, on_msg=on_msg)
    log.info("check done, messages by type: %s", seen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="проверка подключения (20 с, без записи)")
    p.add_argument("--minutes", type=float, help="остановиться через N минут")
    p.add_argument("--tickers", nargs="+", default=TICKERS)
    args = p.parse_args()
    setup_logging()
    if "TINVEST_TOKEN" not in os.environ:
        sys.exit("Задайте токен в переменной окружения TINVEST_TOKEN")
    if args.check:
        check_connection()
        return
    merge_parts()  # дособрать части после аварийного завершения прошлого запуска
    collect(args.tickers, minutes=args.minutes)


if __name__ == "__main__":
    main()
