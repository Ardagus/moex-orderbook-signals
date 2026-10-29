"""Этап 3: признаки (только по данным до t включительно) и цели (доходность mid вперёд).

Все окна считаются внутри куска seg (непрерывные секунды), через дыры данных не перешагиваем.
Неполные окна дают NaN — такие строки на этапе обучения отбрасываются.

Запуск: python features.py   (data/prepared -> data/features)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
IN_DIR = ROOT / "data" / "prepared"
OUT_DIR = ROOT / "data" / "features"
DEPTH = 10
WINDOWS = [5, 30, 60]  # секунды, для OFI / дисбаланса сделок / доходности / волатильности
HORIZONS = [5, 30, 60]  # секунды, на которые смотрим вперёд в цели
# Порог метки θ = 2·(половина спреда (доля от mid, в момент t) + комиссия за одну сторону), то есть полная
# стоимость круга сделки. Правило задано заранее и на тестовых днях не подбирается; комиссия — параметр,
# по умолчанию 0,05% от оборота.
FEE = 0.0005

FEATURE_COLS = (["spread_ticks", "spread_pct", "micro_minus_mid", "imb_1", "imb_5", "imb_10"]
                + [f"{name}_{w}" for name in ("ofi", "tib", "ret", "rv") for w in WINDOWS])
TARGET_COLS = [f"{name}_{h}" for name in ("y", "ysign") for h in HORIZONS]


def infer_tick(df):
    """Шаг цены — статическая характеристика инструмента: минимальная ненулевая разница соседних цен в стакане."""
    prices = df[[f"{side}_px_{i}" for side in ("bid", "ask") for i in range(1, DEPTH + 1)]].to_numpy()
    diffs = np.abs(np.diff(prices, axis=1)).ravel()
    diffs = diffs[diffs > 1e-9]
    return round(float(diffs.min()), 6)  # round убирает шум float: 0.0100000000000193 -> 0.01


def label_threshold(half_spread, fee):
    """θ = 2·(half_spread + fee): вход и выход по половине спреда и комиссии за каждую сторону — полная стоимость круга."""
    return 2 * (half_spread + fee)


def make_labels(y, half_spread, fee):
    """Класс +1 / -1, если доходность mid за горизонт вышла за θ = 2·(half_spread + fee), иначе 0. NaN остаётся NaN."""
    theta = label_threshold(half_spread, fee)
    labels = pd.Series(np.where(y > theta, 1.0, np.where(y < -theta, -1.0, 0.0)), index=y.index)
    return labels.where(y.notna())


def roll_sum(s, seg, w):
    # окно из последних w секунд (включая t), только внутри seg; неполное окно -> NaN
    return s.groupby(seg).rolling(w, min_periods=w).sum().reset_index(level=0, drop=True)


def build_features(df, tick, fee=FEE):
    """df — таблица из prepare.py (один тикер, один день). Возвращает признаки + цели."""
    seg = df.seg
    out = df[["ticker", "ts", "session", "seg"]].copy()

    bid1, ask1 = df.bid_px_1, df.ask_px_1
    mid = (bid1 + ask1) / 2
    out["mid"] = mid  # нужен для цели и бэктеста; как признак не используем (уровень цены не переносится между днями)
    out["bid_px_1"], out["ask_px_1"] = bid1, ask1
    out["half_spread"] = (ask1 - bid1) / mid / 2  # для порога θ; известен в момент t, утечки нет

    # стакан в момент t
    out["spread_ticks"] = (ask1 - bid1) / tick
    out["spread_pct"] = (ask1 - bid1) / mid * 100
    # microprice смещён к стороне с меньшим объёмом: там цена вероятнее сдвинется
    micro = (bid1 * df.ask_qty_1 + ask1 * df.bid_qty_1) / (df.bid_qty_1 + df.ask_qty_1)
    out["micro_minus_mid"] = (micro - mid) / tick
    for k in (1, 5, 10):
        b = df[[f"bid_qty_{i}" for i in range(1, k + 1)]].sum(axis=1)
        a = df[[f"ask_qty_{i}" for i in range(1, k + 1)]].sum(axis=1)
        out[f"imb_{k}"] = (b - a) / (b + a)

    # поток ордеров, сделки «по рынку», прошлая доходность и волатильность
    r1 = mid / mid.groupby(seg).shift(1) - 1  # доходность за 1 с
    for w in WINDOWS:
        out[f"ofi_{w}"] = roll_sum(df.ofi_1, seg, w)
        buy, sell = roll_sum(df.buy_qty, seg, w), roll_sum(df.sell_qty, seg, w)
        out[f"tib_{w}"] = ((buy - sell) / (buy + sell).replace(0, np.nan)).fillna(0).where(buy.notna())  # нет сделок -> 0
        out[f"ret_{w}"] = mid / mid.groupby(seg).shift(w) - 1
        out[f"rv_{w}"] = np.sqrt(roll_sum(r1 ** 2, seg, w))

    # цели: доходность mid на h секунд ВПЕРЁД (это единственное место, где смотрим в будущее)
    for h in HORIZONS:
        out[f"y_{h}"] = mid.groupby(seg).shift(-h) / mid - 1
        out[f"ysign_{h}"] = make_labels(out[f"y_{h}"], out.half_spread, fee)  # 3 класса: -1 / 0 / +1
    return out


def main():
    ticks = {}
    for ticker_dir in sorted(IN_DIR.glob("*")):
        files = sorted(ticker_dir.glob("*.parquet"))
        frames = {f: pd.read_parquet(f) for f in files}
        ticks[ticker_dir.name] = infer_tick(pd.concat(frames.values()))
        (OUT_DIR / ticker_dir.name).mkdir(parents=True, exist_ok=True)
        for f, df in frames.items():
            out = build_features(df, ticks[ticker_dir.name])
            out.to_parquet(OUT_DIR / ticker_dir.name / f.name, index=False)
            ready = out[FEATURE_COLS + TARGET_COLS].notna().all(axis=1).sum()
            print(f"{ticker_dir.name} {f.stem}: строк {len(out)}, полных (признаки+цели) {ready}, шаг цены {ticks[ticker_dir.name]}")


if __name__ == "__main__":
    main()
