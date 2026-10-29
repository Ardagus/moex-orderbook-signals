"""Тесты prepare.py на крошечных ручных примерах."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import prepare  # noqa: E402

SEC = prepare.SEC
# 21:00:00 МСК — внутри вечерней сессии, далеко от срезанных краёв
T0 = pd.Timestamp("2026-10-29 21:00:00+03:00").value


def book_msg(sec, bid, bq, ask, aq, consistent=True):
    """Снимок стакана через sec секунд после T0 (sec может быть дробным)."""
    row = {"kind": "book", "ticker": "X", "exch_ts": int(T0 + sec * SEC), "recv_ts": int(T0 + sec * SEC),
           "is_consistent": consistent}
    for i in range(1, prepare.DEPTH + 1):
        row.update({f"bid_px_{i}": bid - (i - 1), f"bid_qty_{i}": bq, f"ask_px_{i}": ask + (i - 1), f"ask_qty_{i}": aq})
    return row


def trade_msg(sec, direction, qty):
    return {"kind": "trade", "ticker": "X", "exch_ts": int(T0 + sec * SEC), "recv_ts": int(T0 + sec * SEC),
            "trade_px": 100.0, "trade_qty": qty, "trade_dir": f"TRADE_DIRECTION_{direction}"}


def test_ofi_on_hand_example():
    # вручную: 1) размер bid вырос на 5 -> +5; 2) bid поднялся, ask ушёл вверх -> +7+10; 3) bid упал -> -7
    snaps = pd.DataFrame([book_msg(0, 100, 10, 101, 10), book_msg(1, 100, 15, 101, 10),
                          book_msg(2, 101, 7, 102, 5), book_msg(3, 100, 4, 102, 5)])
    assert list(prepare.ofi_events(snaps)) == [0, 5, 17, -7]


def test_ofi_not_glued_across_gap():
    snaps = pd.DataFrame([book_msg(0, 100, 10, 101, 10), book_msg(100, 100, 500, 101, 10)])
    assert list(prepare.ofi_events(snaps)) == [0, 0]  # разрыв 100 с > MAX_BOOK_AGE_S


def test_state_at_t_uses_only_data_up_to_t():
    raw = pd.DataFrame([book_msg(0.0, 100, 10, 101, 10), book_msg(2.5, 105, 10, 106, 10),
                        book_msg(5.0, 105, 10, 106, 10)])
    out, _ = prepare.prepare_day(raw, "X")
    px = out.set_index(out.ts.astype("int64"))["bid_px_1"]
    assert px[T0 + 2 * SEC] == 100  # снимок 2.5 с ещё из будущего для t=2
    assert px[T0 + 3 * SEC] == 105


def test_trades_go_to_interval_t_minus_1_to_t():
    raw = pd.DataFrame([book_msg(0, 100, 10, 101, 10), book_msg(6, 100, 10, 101, 10),
                        trade_msg(2.0, "BUY", 3),    # ровно t=2 -> попадает в секунду 2
                        trade_msg(2.5, "SELL", 4)])  # (2, 3] -> секунда 3
    out, _ = prepare.prepare_day(raw, "X")
    by_sec = out.set_index(out.ts.astype("int64"))
    assert by_sec.loc[T0 + 2 * SEC, ["buy_qty", "sell_qty"]].tolist() == [3, 0]
    assert by_sec.loc[T0 + 3 * SEC, ["buy_qty", "sell_qty"]].tolist() == [0, 4]


def test_stale_book_and_bad_snapshots_are_dropped():
    raw = pd.DataFrame([book_msg(0, 100, 10, 101, 10), book_msg(60, 100, 10, 101, 10),
                        book_msg(61, 100, 10, 101, 10, consistent=False)])
    out, stats = prepare.prepare_day(raw, "X")
    assert stats["bad_snapshots"] == 1
    assert out.seg.nunique() == 2  # секунды 31..59 с устаревшим стаканом выброшены -> два куска
    assert out.book_age_s.max() <= prepare.MAX_BOOK_AGE_S


def test_session_edges_and_outside_hours_are_dropped():
    # 19:05 МСК — начало вечерней сессии: первые TRIM_MIN минут срезаются
    start = pd.Timestamp("2026-10-29 19:05:00+03:00").value
    before_open = pd.Timestamp("2026-10-29 09:00:00+03:00").value  # до начала основной сессии при любом расписании
    t = pd.Series([start, start + 4 * 60 * SEC, start + 5 * 60 * SEC, before_open], dtype="int64")
    labels = prepare.session_labels(t.to_numpy())
    assert list(labels) == [None, None, "evening", None]  # 19:05, 19:09 — срез; 19:10 — внутри; 09:00 — вне торгов
