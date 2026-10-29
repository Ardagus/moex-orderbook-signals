"""Тесты backtest.py: цены исполнения, задержка, удержание, издержки, пометки синтетики."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import backtest as B  # noqa: E402


def market_and_preds(mid, spread, cls, h, n=None):
    """Рынок с заданным mid и полным спредом spread (bid = mid - spread/2, ask = mid + spread/2) + сигнал cls по секундам."""
    n = len(mid)
    ts = pd.date_range("2026-01-05 10:00", periods=n, freq="s", tz="UTC")
    mid = np.asarray(mid, dtype=float)
    m = pd.DataFrame({"ticker": "X", "ts": ts, "mid": mid, "bid_px_1": mid - spread / 2, "ask_px_1": mid + spread / 2})
    p = pd.DataFrame({"ticker": "X", "ts": ts, "day": "d0", "imb1_cls": cls, "half_spread": spread / 2 / mid})
    return m, {("X", h): p}


def run(m, preds, delay, h=None):
    return B.run_backtest(m, preds, ["cls_imb1"], [delay], signal_fee=0.0005)


def test_long_buys_at_ask_and_sells_at_bid_with_delay():
    mid = 100 + np.arange(12.0)  # растёт на 1 в секунду
    cls = np.zeros(12, int)
    cls[0] = 1
    m, preds = market_and_preds(mid, 1.0, cls, h=3)  # bid = mid-0.5, ask = mid+0.5
    tr = run(m, preds, delay=1)
    t = tr.iloc[0]
    assert len(tr) == 1 and t.entry_t - B.to_sec(m.ts)[0] == 1 and t.exit_t - t.entry_t == 3  # вход t+1, выход через h
    assert t.entry_px == 101.5 and t.exit_px == 103.5  # ask в t=1, bid в t=4
    assert np.isclose(t.exec_ret, 103.5 / 101.5 - 1) and np.isclose(t.gross_ret, 104 / 101 - 1)


def test_short_sells_at_bid_and_buys_back_at_ask():
    mid = 100 + np.arange(12.0)
    cls = np.zeros(12, int)
    cls[0] = -1
    m, preds = market_and_preds(mid, 1.0, cls, h=3)
    t = run(m, preds, delay=1).iloc[0]
    assert t.entry_px == 100.5 and t.exit_px == 104.5  # bid в t=1, ask в t=4
    assert np.isclose(t.exec_ret, -(104.5 / 100.5 - 1))


def test_delay_shifts_execution_time():
    cls = np.zeros(15, int)
    cls[0] = 1
    m, preds = market_and_preds(100 + np.arange(15.0), 1.0, cls, h=3)
    for delay in (0, 1, 3, 5):
        t = run(m, preds, delay).iloc[0]
        assert t.entry_t - B.to_sec(m.ts)[0] == delay


def test_same_signal_extends_and_opposite_signal_reverses():
    cls = np.zeros(20, int)
    cls[[0, 2]] = 1  # второй +1 продлевает удержание до 2+3=5
    cls[5] = -1  # на t=5 длинная позиция уже должна закрыться и открыться короткая
    m, preds = market_and_preds(np.full(20, 100.0), 0.02, cls, h=3)
    tr = run(m, preds, delay=0).sort_values("entry_t").reset_index(drop=True)
    t0 = B.to_sec(m.ts)[0]
    assert tr[["side", "entry_t", "exit_t"]].values.tolist() == [[1, t0, t0 + 5], [-1, t0 + 5, t0 + 8]]
    # разворот раньше горизонта: +1 на 0, -1 на 2 -> лонг 0..2, шорт 2..5
    cls2 = np.zeros(20, int)
    cls2[0], cls2[2] = 1, -1
    m2, preds2 = market_and_preds(np.full(20, 100.0), 0.02, cls2, h=3)
    tr2 = run(m2, preds2, delay=0).sort_values("entry_t")
    assert tr2[["side", "entry_t", "exit_t"]].values.tolist() == [[1, t0, t0 + 2], [-1, t0 + 2, t0 + 5]]


def test_no_trade_when_execution_row_is_missing():
    t = np.array([0, 2, 3, 4, 5])  # секунды 1 нет
    sig = np.array([1, 0, 0, 0, 0])
    assert B.simulate_day(t, sig, h=2, delay=1) == []
    assert len(B.simulate_day(t, sig, h=2, delay=2)) == 1


def test_spread_is_paid_once_per_round_trip_not_twice():
    cls = np.where(np.arange(60) % 10 == 0, 1, 0)
    m, preds = market_and_preds(np.full(60, 100.0), 0.02, cls, h=3)  # спред 2 б.п. от цены
    tr = run(m, preds, delay=1)
    # покупка по ask и продажа по bid: потеря = ровно один полный спред (0.02%) = две половины
    assert np.allclose(tr.exec_ret, 99.99 / 100.01 - 1) and np.allclose(tr.gross_ret, 0)
    # без спреда и без комиссии стратегия на плоском рынке ровно в нуле
    m0, preds0 = market_and_preds(np.full(60, 100.0), 0.0, cls, h=3)
    s = B.summarize(run(m0, preds0, delay=1), fee=0.0, all_days=["d0"])
    assert s["total_net_bp"] == 0 and s["breakeven_fee_pct"] == 0


def test_zero_information_signal_with_commission_has_strictly_negative_pnl():
    rng = np.random.default_rng(0)
    n = 600
    cls = rng.choice([-1, 0, 1], n)  # сигнал без информации; рынок плоский, значит валовый PnL = 0
    for spread in (0.0, 0.02):
        m, preds = market_and_preds(np.full(n, 100.0), spread, cls, h=5)
        tr = run(m, preds, delay=1)
        assert len(tr) > 50
        s = B.summarize(tr, fee=0.0005, all_days=["d0"])
        assert s["total_gross_bp"] == 0 and s["total_net_bp"] < 0 and s["avg_net_bp"] < 0
        assert s["hit_net"] == 0 and s["breakeven_fee_pct"] <= 0


def test_zero_information_signal_on_random_walk_loses_costs():
    rng = np.random.default_rng(1)
    n = 3000
    mid = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.5e-4, n))), 2)
    m, preds = market_and_preds(mid, 0.02, rng.choice([-1, 1], n), h=5)
    s = B.summarize(run(m, preds, delay=1), fee=0.0005, all_days=["d0"])
    assert s["total_net_bp"] < 0 and abs(s["avg_gross_bp"]) < 0.5


def test_synthetic_planted_signal_is_found_and_dies_with_fee():
    market, preds = B.make_synthetic(seed=1, n_days=3, secs=1200)
    tr = B.run_backtest(market, preds, ["reg_ridge", "cls_imb1"], [0, 5], signal_fee=0.0005)
    days = {h: sorted({d for (tk, hh), p in preds.items() if hh == h for d in p.day.unique()}) for h in (30, 60)}
    ridge = tr[(tr.signal == "reg_ridge") & (tr.h == 30) & (tr.delay == 0)]
    assert B.summarize(ridge, 0.0, days[30])["total_gross_bp"] > 0  # заложенный сигнал даёт валовую прибыль
    nets = [B.summarize(ridge, f, days[30])["avg_net_bp"] for f in B.FEES]
    assert all(a > b for a, b in zip(nets, nets[1:]))  # чем выше комиссия, тем хуже
    noise = tr[(tr.signal == "cls_imb1") & (tr.h == 30) & (tr.delay == 0)]
    assert B.summarize(noise, 0.0005, days[30])["total_net_bp"] < 0  # случайные знаки на комиссии проигрывают


def test_drawdown_and_sharpe_on_hand_examples():
    assert B.max_drawdown([1, -2, 3, -1, -1]) == 2  # кривая 0,1,-1,2,1,0: пики 1 и 2, просадка 2 и 2
    assert B.max_drawdown([1, 2, 3]) == 0
    assert np.isclose(B.sharpe_daily([1, 2, 3]), 2 * np.sqrt(252))  # mean 2, std 1
    assert np.isnan(B.sharpe_daily([5.0])) and np.isnan(B.sharpe_daily([1, 1, 1]))


def test_non_real_results_are_marked_on_figures_and_banner():
    for kind in ("synthetic", "pipeline_check_pseudo_days"):
        fig = plt.figure()
        B.add_marks(fig, "заголовок", kind)
        texts = [t.get_text() for t in fig.texts]
        assert "SYNTHETIC / PIPELINE CHECK" in texts and "[SYNTHETIC / PIPELINE CHECK]" in fig._suptitle.get_text()
        assert "SYNTHETIC / PIPELINE CHECK" in B.banner(kind)
        plt.close(fig)
    fig = plt.figure()
    B.add_marks(fig, "заголовок", "real")
    assert not any("SYNTHETIC" in t.get_text() for t in fig.texts)  # у реальных данных пометок нет
    plt.close(fig)
    assert B.data_kind({"block_minutes": 3}) == "pipeline_check_pseudo_days" and B.data_kind({"block_minutes": None}) == "real"
