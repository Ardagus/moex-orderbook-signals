"""Тесты features.py: утечка из будущего, сдвиг цели, ручные примеры."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import features  # noqa: E402

TICK = 0.01


def make_df(n=300, seed=0, seg=None):
    """Синтетическая таблица в формате prepare.py: случайное блуждание mid, случайные объёмы."""
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.integers(-1, 2, n)) * TICK
    d = {"ticker": "X", "ts": pd.date_range("2026-10-29 18:00", periods=n, freq="s", tz="UTC"),
         "session": "evening", "seg": 1 if seg is None else seg}
    for i in range(1, features.DEPTH + 1):
        d[f"bid_px_{i}"] = np.round(mid - TICK / 2 - (i - 1) * TICK, 2)
        d[f"ask_px_{i}"] = np.round(mid + TICK / 2 + (i - 1) * TICK, 2)
        d[f"bid_qty_{i}"] = rng.integers(1, 50, n)
        d[f"ask_qty_{i}"] = rng.integers(1, 50, n)
    d.update(buy_qty=rng.integers(0, 20, n), sell_qty=rng.integers(0, 20, n), ofi_1=rng.normal(0, 10, n))
    return pd.DataFrame(d)


def test_features_at_t_do_not_depend_on_data_after_t():
    df = make_df()
    k = 150
    changed = df.copy()
    rng = np.random.default_rng(1)
    for c in changed.columns:  # портим ВСЁ после момента k: цены, объёмы, поток
        if c.startswith(("bid_", "ask_")) or c in ("buy_qty", "sell_qty", "ofi_1"):
            changed[c] = changed[c].astype(float)
            changed.loc[k + 1:, c] = changed.loc[k + 1:, c] * rng.uniform(0.5, 3, len(changed) - k - 1)
    cols = ["mid"] + features.FEATURE_COLS
    a = features.build_features(df, TICK).loc[:k, cols]
    b = features.build_features(changed, TICK).loc[:k, cols]
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_target_is_forward_shift_and_stays_inside_segment():
    seg = np.r_[np.full(100, 1), np.full(100, 2)]
    df = make_df(200, seg=seg)
    out = features.build_features(df, TICK)
    mid = out.mid.to_numpy()
    assert np.isclose(out.y_5[10], mid[15] / mid[10] - 1)  # значение через ровно 5 строк = 5 секунд
    assert np.isclose(out.y_60[0], mid[60] / mid[0] - 1)
    assert out.y_5[95:100].isna().all()  # последние 5 строк куска не заглядывают в следующий кусок
    assert np.isclose(out.y_5[100], mid[105] / mid[100] - 1)
    expected = features.make_labels(out.y_5, out.half_spread, features.FEE)
    pd.testing.assert_series_equal(out.ysign_5, expected, check_names=False)


def test_label_threshold_is_full_round_trip_cost():
    # θ = 2·(half_spread + fee) = 2·(0.0001 + 0.0005) = 0.0012: вход и выход, у каждого половина спреда и комиссия
    assert np.isclose(features.label_threshold(0.0001, 0.0005), 0.0012)
    assert np.isclose(features.label_threshold(0.0001, 0.0), 0.0002)


def test_make_labels_threshold_is_twice_half_spread_plus_fee():
    hs = pd.Series(0.0001, index=range(8))
    theta = features.label_threshold(hs, 0.0005)  # 0.0012; на самом пороге метка 0 (неравенства строгие)
    y = pd.Series([0.0013, -0.0013, 0.0012, -0.0012, 0.0007, 0.0006, 0.0, np.nan])  # 0.0006 — прежний порог θ = hs + fee
    labels = features.make_labels(y, hs, fee=0.0005)
    assert labels.iloc[:7].tolist() == [1, -1, 0, 0, 0, 0, 0] and np.isnan(labels.iloc[7])
    assert np.isclose(theta.iloc[0], 0.0012)
    # без комиссии порог уже (0.0002): y=0.0007 становится сигналом, на самом пороге метка по-прежнему 0
    no_fee = features.make_labels(y, hs, fee=0.0)
    assert no_fee.iloc[4] == 1 and features.make_labels(pd.Series([0.0002]), hs.iloc[:1], fee=0.0).iloc[0] == 0


def test_hand_example_spread_microprice_imbalance():
    df = make_df(70)
    df["bid_px_1"], df["ask_px_1"] = 100.00, 100.02
    df["bid_qty_1"], df["ask_qty_1"] = 10, 30
    for i in range(2, 11):
        df[f"bid_qty_{i}"], df[f"ask_qty_{i}"] = 10, 10
    row = features.build_features(df, TICK).iloc[0]
    assert np.isclose(row.spread_ticks, 2)
    assert np.isclose(row.micro_minus_mid, -0.5)  # micro = (100*30 + 100.02*10)/40 = 100.005, mid = 100.01
    assert np.isclose(row.imb_1, -0.5)  # (10-30)/(10+30)
    assert np.isclose(row.imb_5, -20 / 120)  # bid 50, ask 30+40=70
    assert np.isclose(row.imb_10, (100 - 120) / 220)


def test_rolling_windows_are_full_only_and_reset_between_segments():
    seg = np.r_[np.full(40, 1), np.full(40, 2)]
    df = make_df(80, seg=seg)
    df["ofi_1"] = 1.0
    out = features.build_features(df, TICK)
    assert out.ofi_5[:4].isna().all() and (out.ofi_5[4:40] == 5).all()
    assert out.ofi_5[40:44].isna().all() and out.ofi_5[44] == 5  # во втором куске окно копится заново


def test_trade_imbalance_zero_without_trades():
    df = make_df(70)
    df["buy_qty"], df["sell_qty"] = 0, 0
    out = features.build_features(df, TICK)
    assert (out.tib_5[4:] == 0).all()
    df["buy_qty"], df["sell_qty"] = 3, 1
    assert np.isclose(features.build_features(df, TICK).tib_5[10], (15 - 5) / (15 + 5))


def test_infer_tick():
    assert features.infer_tick(make_df()) == TICK
