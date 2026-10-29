"""Этап 2: сырой поток -> равномерная сетка 1 с (data/prepared/{ticker}/{date}.parquet).

Время везде биржевое (exch_ts), чтобы не зависеть от сдвига часов ПК. Состояние на момент t —
последний снимок стакана с exch_ts <= t; сделки и OFI за секунду t — те, что в интервале (t-1, t].

Запуск: python prepare.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "prepared"
DEPTH = 10
SEC = 1_000_000_000  # наносекунд в секунде
MAX_BOOK_AGE_S = 30  # стакан старше — считаем разрывом данных, секунду выкидываем
TRIM_MIN = 5  # срезаем первые и последние минуты каждой сессии
# непрерывные торги акций, МСК. Источник — страница MOEX 2022 г., на 2026 г. не перепроверено
SESSIONS = {"main": ("10:00", "19:00"), "evening": ("19:05", "23:50")}
MSK_OFFSET_NS = 3 * 3600 * SEC

LEVEL_COLS = [f"{side}_{kind}_{i}" for i in range(1, DEPTH + 1) for side in ("bid", "ask") for kind in ("px", "qty")]


def clean_books(book):
    """Оставляем только надёжные снимки: консистентные, с обеими сторонами, без пересечения."""
    ok = (book["is_consistent"].astype(bool) & book.bid_px_1.notna() & book.ask_px_1.notna()
          & (book.bid_px_1 < book.ask_px_1))
    return book[ok].sort_values(["exch_ts", "recv_ts"], kind="stable").reset_index(drop=True), int((~ok).sum())


def ofi_events(book, max_gap_s=MAX_BOOK_AGE_S):
    """OFI по Cont et al. для лучшего уровня: вклад каждого изменения стакана относительно предыдущего снимка."""
    pb, qb, pa, qa = (book[c] for c in ("bid_px_1", "bid_qty_1", "ask_px_1", "ask_qty_1"))
    pb0, qb0, pa0, qa0 = pb.shift(), qb.shift(), pa.shift(), qa.shift()
    e = (np.where(pb >= pb0, qb, 0) - np.where(pb <= pb0, qb0, 0)
         - np.where(pa <= pa0, qa, 0) + np.where(pa >= pa0, qa0, 0))
    gap = book.exch_ts.diff()
    # через разрыв данных события не склеиваем: иначе получится выдуманный огромный "поток"
    e = np.where(gap.isna() | (gap > max_gap_s * SEC), 0, e)
    return pd.Series(e, index=book.index, dtype="float64")


def ceil_sec(ns):
    """Секунда, в интервал (t-1, t] которой попадает момент ns."""
    return -(-ns // SEC) * SEC


def session_labels(ts_ns):
    """Метка сессии для каждой секунды (None, если вне окон торгов или в срезанных краях)."""
    tod = ((ts_ns + MSK_OFFSET_NS) % (24 * 3600 * SEC)) // SEC  # секунд от полуночи по МСК
    labels = np.full(len(ts_ns), None, dtype=object)
    for name, (start, end) in SESSIONS.items():
        s = pd.Timedelta(f"{start}:00").seconds + TRIM_MIN * 60
        e = pd.Timedelta(f"{end}:00").seconds - TRIM_MIN * 60
        labels[(tod >= s) & (tod <= e)] = name
    return labels


def prepare_day(raw, ticker):
    """raw — сырая таблица одного тикера за день (book + trade). Возвращает (таблица, статистика отбраковки)."""
    if "trade_dir" not in raw:  # за день не было ни одной сделки -> колонок сделок нет
        raw = raw.assign(trade_dir=None, trade_qty=0)
    book, n_bad = clean_books(raw[raw.kind == "book"])
    trades = raw[raw.kind == "trade"]
    stats = {"snapshots": len(book) + n_bad, "bad_snapshots": n_bad}
    if book.empty:
        return pd.DataFrame(), stats

    t_start, t_end = ceil_sec(book.exch_ts.iloc[0]), book.exch_ts.iloc[-1] // SEC * SEC
    grid = pd.DataFrame({"ts_ns": np.arange(t_start, t_end + 1, SEC, dtype="int64")})

    # состояние стакана на момент t: последний снимок не позже t (backward — будущее не подглядываем)
    cols = ["exch_ts"] + LEVEL_COLS
    grid = pd.merge_asof(grid, book[cols], left_on="ts_ns", right_on="exch_ts", direction="backward")
    grid["book_age_s"] = (grid.ts_ns - grid.exch_ts) / SEC
    grid = grid.drop(columns="exch_ts")

    # сделки и OFI, агрегированные по секундам (t-1, t]
    tr = trades.assign(sec=ceil_sec(trades.exch_ts))
    for side, name in (("TRADE_DIRECTION_BUY", "buy"), ("TRADE_DIRECTION_SELL", "sell")):
        part = tr[tr.trade_dir == side].groupby("sec").trade_qty.agg(["sum", "size"])
        grid[f"{name}_qty"] = grid.ts_ns.map(part["sum"]).fillna(0)
        grid[f"n_{name}"] = grid.ts_ns.map(part["size"]).fillna(0).astype(int)
    ofi = ofi_events(book).groupby(ceil_sec(book.exch_ts)).sum()
    grid["ofi_1"] = grid.ts_ns.map(ofi).fillna(0)

    # фильтры: вне сессии / края сессии, разрыв данных (старый стакан)
    grid["session"] = session_labels(grid.ts_ns.to_numpy())
    in_session = grid.session.notna()
    fresh = grid.book_age_s <= MAX_BOOK_AGE_S
    stats.update(grid_seconds=len(grid), in_session=int(in_session.sum()), stale=int((in_session & ~fresh).sum()))
    grid = grid[in_session & fresh].reset_index(drop=True)

    # seg — номер непрерывного куска секунд; признаки и цель потом считаем внутри куска, не через дыры
    grid["seg"] = (grid.ts_ns.diff() != SEC).cumsum()
    grid.insert(0, "ts", pd.to_datetime(grid.pop("ts_ns"), unit="ns", utc=True))
    grid.insert(0, "ticker", ticker)
    stats["kept"] = len(grid)
    return grid, stats


def main():
    for path in sorted(RAW_DIR.glob("*/*.parquet")):
        ticker, day = path.parent.name, path.stem
        out, stats = prepare_day(pd.read_parquet(path), ticker)
        if out.empty:
            print(f"{ticker} {day}: нет данных внутри сессии {stats}")
            continue
        (OUT_DIR / ticker).mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT_DIR / ticker / f"{day}.parquet", index=False)
        print(f"{ticker} {day}: {stats}, сегментов: {out.seg.nunique()}")


if __name__ == "__main__":
    main()
