"""Этап 5: бэктест с издержками на прогнозах train.py (или на синтетике для проверки пайплайна).

Правила:
  * сигнал (-1/0/+1) строится в момент t по прогнозу; исполнение на сетке в t + delay секунд;
  * покупка по ask, продажа по bid из стакана в момент исполнения (спред учитывается ЗДЕСЬ, один раз;
    порог θ в самом сигнале лишь решает, торговать ли, и в PnL не вычитается);
  * позиция ±1 лот; тот же сигнал продлевает удержание, противоположный — разворот, иначе выход через h секунд;
  * комиссия — доля от оборота за сторону, считается после симуляции (сетка 0 / 0,02 / 0,05 / 0,1%).

Результаты ВСЕГДА кладутся в results/backtest/<вид данных>/ и помечены на графиках и в таблицах:
  synthetic | pipeline_check_pseudo_days | real

Запуск: python backtest.py            (прогнозы из data/predictions)
        python backtest.py --synthetic  (синтетический рынок с заложенным сигналом)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: сохраняем только в файлы
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import features as F  # noqa: E402

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "data" / "predictions"
OUT_ROOT = ROOT / "results" / "backtest"
DELAYS = [0, 1, 3, 5]  # секунды между сигналом и исполнением
FEES = [0.0, 0.0002, 0.0005, 0.001]  # комиссия за сторону, доля от оборота
SIGNALS = ["cls_lgbm", "cls_logit", "cls_imb1", "reg_ridge", "reg_lgbm"]
MAIN_HORIZONS = [30, 60]
CONTROL = "cls_imb1"  # правило «знак imbalance_1»: бейзлайн из ТЗ, на графиках накопленного PnL показывается отдельно
BP = 1e4

KIND_NOTE = {"synthetic": "синтетический рынок", "pipeline_check_pseudo_days": "реальные данные, нарезанные на псевдо-дни",
             "real": "реальные данные"}
# палитра: категориальные слоты 1-5 (светлая тема) закреплены за сигналами, чтобы цвет следовал сущности
SIGNAL_COLOR = {"cls_lgbm": "#2a78d6", "cls_logit": "#eb6834", "cls_imb1": "#1baf7a", "reg_ridge": "#eda100",
                "reg_lgbm": "#e87ba4"}
DELAY_COLOR = {0: "#86b6ef", 1: "#5598e7", 3: "#2a78d6", 5: "#184f95"}  # порядковая шкала: больше задержка — темнее
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ---------- сигнал и симуляция ----------

def to_sec(ts):
    return ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()


def make_signal(preds, name, signal_fee):
    """cls_* — готовые классы; reg_* — прогноз доходности против того же порога θ = 2·(half_spread + fee)."""
    kind, model = name.split("_")
    if kind == "cls":
        return preds[f"{model}_cls"].fillna(0).to_numpy().astype(int)
    theta = F.label_threshold(preds.half_spread, signal_fee).to_numpy()
    p = preds[f"{model}_y"].to_numpy()
    return np.where(p > theta, 1, np.where(p < -theta, -1, 0))


def last_row(t, time):
    return int(np.searchsorted(t, time, side="right")) - 1  # последняя строка со временем <= time


def simulate_day(t, sig, h, delay):
    """t — секунды (по возрастанию), sig — сигнал в каждую секунду. Возвращает сделки (строка входа, строка выхода, сторона)."""
    trades, pos, entry_i, exit_t = [], 0, -1, 0
    for i in np.flatnonzero(sig):
        te = t[i] + delay
        i_e = int(np.searchsorted(t, te))
        if i_e >= len(t) or t[i_e] != te:
            continue  # в момент исполнения в данных дыра — не торгуем
        if pos != 0 and exit_t <= te:  # горизонт удержания истёк до этого сигнала
            trades.append((entry_i, last_row(t, exit_t), pos))
            pos = 0
        s = int(sig[i])
        if pos == s:
            exit_t = te + h  # тот же сигнал: продлеваем удержание
        else:
            if pos != 0:
                trades.append((entry_i, i_e, pos))  # противоположный сигнал: закрываем и разворачиваемся
            pos, entry_i, exit_t = s, i_e, te + h
    if pos != 0:
        trades.append((entry_i, last_row(t, exit_t), pos))  # в конце данных/горизонта закрываем по последней цене
    return trades


def trade_frame(trades, t, bid, ask, mid):
    ei, xi, side = (np.array(c) for c in zip(*trades))
    long_ = side > 0
    return pd.DataFrame({
        "side": side, "entry_t": t[ei], "exit_t": t[xi],
        "entry_px": np.where(long_, ask[ei], bid[ei]),  # худшая сторона: покупка по ask, продажа по bid
        "exit_px": np.where(long_, bid[xi], ask[xi]),
        "entry_mid": mid[ei], "exit_mid": mid[xi]})


def run_backtest(market, preds, signals, delays, signal_fee):
    """market: ticker, ts, bid_px_1, ask_px_1, mid. preds: {(ticker, h): таблица прогнозов с колонкой day}."""
    mk = {tk: g.sort_values("ts") for tk, g in market.groupby("ticker")}
    chunks = []
    for (ticker, h), p in sorted(preds.items()):
        m = mk[ticker]
        m_t = to_sec(m.ts)
        for day, pday in p.groupby("day"):
            p_t = to_sec(pday.ts)
            win = (m_t >= p_t.min()) & (m_t <= p_t.max() + h + max(delays))  # с запасом на выходы после конца дня
            t, bid, ask, mid = m_t[win], m.bid_px_1.to_numpy()[win], m.ask_px_1.to_numpy()[win], m.mid.to_numpy()[win]
            idx = np.searchsorted(t, p_t)
            assert (t[idx] == p_t).all(), "строки прогноза должны быть в таблице рынка"
            for name in signals:
                s = make_signal(pday, name, signal_fee)
                sig = np.zeros(len(t), dtype=int)
                sig[idx] = s
                for delay in delays:
                    trades = simulate_day(t, sig, h, delay)
                    if trades:
                        chunks.append(trade_frame(trades, t, bid, ask, mid).assign(
                            signal=name, h=h, delay=delay, ticker=ticker, day=day))
    if not chunks:
        return pd.DataFrame()
    tr = pd.concat(chunks, ignore_index=True)
    tr["gross_ret"] = tr.side * (tr.exit_mid / tr.entry_mid - 1)  # по mid, без спреда и комиссии
    tr["exec_ret"] = tr.side * (tr.exit_px / tr.entry_px - 1)  # по ценам стакана (спред учтён), без комиссии
    tr["fee_base"] = 1 + tr.exit_px / tr.entry_px  # комиссия = fee * fee_base (за вход и за выход)
    return tr


# ---------- метрики ----------

def max_drawdown(pnl):
    cum = np.concatenate([[0.0], np.cumsum(pnl)])
    return float((np.maximum.accumulate(cum) - cum).max())


def sharpe_daily(daily):
    daily = np.asarray(daily, dtype=float)
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return np.nan
    return daily.mean() / daily.std(ddof=1) * np.sqrt(252)


def summarize(tr, fee, all_days):
    """Метрики по набору сделок при комиссии fee. Все PnL в б.п. от цены входа; 1 лот."""
    if tr.empty:
        return {"n_trades": 0}
    net = (tr.exec_ret - fee * tr.fee_base) * BP
    gross, exe = tr.gross_ret * BP, tr.exec_ret * BP
    daily = net.groupby(tr.day).sum().reindex(all_days, fill_value=0.0)  # дни без сделок = 0
    order = np.argsort(tr.exit_t.to_numpy(), kind="stable")
    return {"n_trades": len(tr), "avg_gross_bp": gross.mean(), "avg_after_spread_bp": exe.mean(), "avg_net_bp": net.mean(),
            "total_gross_bp": gross.sum(), "total_net_bp": net.sum(),
            "hit_gross": (gross > 0).mean(), "hit_net": (net > 0).mean(),
            "sharpe_daily": sharpe_daily(daily), "n_days": len(all_days),
            "max_dd_bp": max_drawdown(net.to_numpy()[order]),
            # комиссия за сторону, при которой средний PnL на сделку обнуляется (<0: убыточно даже без комиссии)
            "breakeven_fee_pct": tr.exec_ret.mean() / tr.fee_base.mean() * 100}


def summary_tables(tr, day_sets, group_by_ticker=False):
    keys = ["signal", "h", "delay"] + (["ticker"] if group_by_ticker else [])
    rows = []
    for key, g in tr.groupby(keys):
        for fee in FEES:
            rows.append({**dict(zip(keys, key)), "fee_pct": fee * 100, **summarize(g, fee, day_sets[key[1]])})
    return pd.DataFrame(rows)


# ---------- рынок для проверки пайплайна ----------

def make_synthetic(seed=0, n_days=3, secs=1500, tickers=("SYN1", "SYN2"), horizons=(30, 60)):
    """Рынок с заложенным сигналом: дрейф mid пропорционален скрытому AR(1)-процессу s_t, который «знает» прогноз.
    Спред 2 тика (1 б.п. на сторону). imb1_cls — случайные знаки, то есть сигнал без информации."""
    rng = np.random.default_rng(seed)
    phi, mu, sigma = 0.95, 0.5e-4, 0.6e-4  # персистентность сигнала, дрейф и шум за секунду (доли)
    market, preds = [], {}
    for tk in tickers:
        parts = {h: [] for h in horizons}
        for d in range(n_days):
            n = secs
            s = np.zeros(n)
            eps = rng.normal(size=n) * np.sqrt(1 - phi ** 2)
            for i in range(1, n):
                s[i] = phi * s[i - 1] + eps[i]
            logret = np.zeros(n)
            logret[1:] = mu * s[:-1] + rng.normal(0, sigma, n - 1)  # доходность за (t, t+1] зависит от s_t
            mid = np.round(100 * np.exp(np.cumsum(logret)), 2)
            ts = pd.date_range(f"2026-01-{5 + d:02d} 10:00", periods=n, freq="s", tz="UTC")
            m = pd.DataFrame({"ticker": tk, "ts": ts, "mid": mid, "bid_px_1": np.round(mid - 0.01, 2),
                              "ask_px_1": np.round(mid + 0.01, 2)})
            m["half_spread"] = (m.ask_px_1 - m.bid_px_1) / m.mid / 2
            market.append(m)
            for h in horizons:
                expected = mu * s * (1 - phi ** h) / (1 - phi)  # E[y_h | s_t]
                p = m[["ticker", "ts", "half_spread"]].copy()
                p["day"] = f"syn{d}"
                p["ridge_y"] = expected + rng.normal(0, 1e-4, n)
                p["lgbm_y"] = expected + rng.normal(0, 2e-4, n)
                theta = F.label_threshold(p.half_spread, F.FEE)
                for model, col in (("lgbm", "lgbm_y"), ("logit", "ridge_y")):
                    p[f"{model}_cls"] = np.where(p[col] > theta, 1, np.where(p[col] < -theta, -1, 0))
                p["imb1_cls"] = rng.choice([-1, 1], n)
                parts[h].append(p.iloc[: n - h])  # последние h секунд без цели не торгуем
        for h in horizons:
            preds[(tk, h)] = pd.concat(parts[h], ignore_index=True)
    return pd.concat(market, ignore_index=True), preds


def load_real():
    meta = json.loads((PRED_DIR / "_meta.json").read_text(encoding="utf-8"))
    market = pd.concat([pd.read_parquet(f) for f in sorted(F.OUT_DIR.glob("*/*.parquet"))], ignore_index=True)
    market = market[["ticker", "ts", "bid_px_1", "ask_px_1", "mid", "half_spread"]]
    preds = {}
    for f in sorted(PRED_DIR.glob("*_h*.parquet")):
        ticker, h = f.stem.rsplit("_h", 1)
        preds[(ticker, int(h))] = pd.read_parquet(f).merge(market[["ticker", "ts", "half_spread"]], on=["ticker", "ts"])
    return market, preds, meta


def data_kind(meta):
    return "pipeline_check_pseudo_days" if meta.get("block_minutes") else "real"


def banner(kind, n_days=None):
    if kind == "real":
        extra = f" (дней: {n_days}; по ТЗ нужно >= 5, иначе результат предварительный)" if n_days and n_days < 5 else ""
        return f"РЕАЛЬНЫЕ ДАННЫЕ{extra}"
    return f"!!! SYNTHETIC / PIPELINE CHECK — {KIND_NOTE[kind]}; НЕ РЕАЛЬНЫЙ РЕЗУЛЬТАТ !!!"


# ---------- графики ----------

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def add_marks(fig, title, kind):
    fig.patch.set_facecolor(SURFACE)
    prefix = "" if kind == "real" else "[SYNTHETIC / PIPELINE CHECK] "
    fig.suptitle(prefix + title, color=INK, fontsize=12, x=0.01, ha="left")
    if kind != "real":  # метка на каждом графике, чтобы его нельзя было принять за настоящий результат
        fig.text(0.5, 0.5, "SYNTHETIC / PIPELINE CHECK", fontsize=36, color="#d03b3b", alpha=0.14, rotation=20,
                 ha="center", va="center")
        fig.text(0.5, 0.005, f"НЕ РЕАЛЬНЫЕ РЕЗУЛЬТАТЫ — {KIND_NOTE[kind]}", ha="center", color="#d03b3b", fontsize=9)


def finish(fig, title, kind, path):
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))  # место под заголовок сверху и плашку «не результат» снизу
    add_marks(fig, title, kind)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_cum_pnl(tr, h, delay, fee, kind, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    d = tr[(tr.h == h) & (tr.delay == delay)]
    empty = []
    for name in SIGNALS:
        g = d[d.signal == name].sort_values("exit_t")
        if g.empty:
            empty.append(name)
            continue
        net = (g.exec_ret - fee * g.fee_base) * BP
        if name == CONTROL:
            continue  # контроль без информации торгует каждую секунду и своим масштабом глушит остальные — см. подпись
        for ax, ret in zip(axes, (g.gross_ret * BP, net)):
            y = np.cumsum(ret.to_numpy())
            ax.plot(np.arange(1, len(y) + 1), y, color=SIGNAL_COLOR[name], linewidth=1.6, label=name)
    ctl = d[d.signal == CONTROL]
    note = f"{CONTROL} (контроль без информации) вне масштаба: n={len(ctl)}, "
    note += f"чистый итог {((ctl.exec_ret - fee * ctl.fee_base) * BP).sum():.0f} б.п." if len(ctl) else "сделок нет"
    for ax, ttl in zip(axes, ("валовый (по mid, без спреда и комиссии)", f"чистый (bid/ask + комиссия {fee:.2%}/сторона)")):
        style_ax(ax)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(ttl, color=INK2, fontsize=10, loc="left")
        ax.set_xlabel("номер сделки (все тикеры, по времени выхода)", color=INK2, fontsize=9)
    axes[0].set_ylabel("накопленный PnL, б.п. от цены входа (1 лот)", color=INK2, fontsize=9)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].text(0.01, 0.99, note, transform=axes[0].transAxes, ha="left", va="top", color=MUTED, fontsize=8)
    if empty:
        axes[1].text(0.01, 0.99, "без сделок: " + ", ".join(empty), transform=axes[1].transAxes, ha="left", va="top",
                     color=MUTED, fontsize=8)
    finish(fig, f"Накопленный PnL, горизонт {h} с, задержка {delay} с", kind, path)


def plot_sensitivity(summary, h, kind, path):
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(3.3 * len(SIGNALS), 3.8), sharey=True)
    s = summary[summary.h == h]
    for ax, name in zip(axes, SIGNALS):
        style_ax(ax)
        ax.axhline(0, color=MUTED, linewidth=1)
        g = s[s.signal == name]
        for delay in DELAYS:
            gd = g[g.delay == delay].sort_values("fee_pct")
            if not gd.empty:
                ax.plot(gd.fee_pct, gd.avg_net_bp, color=DELAY_COLOR[delay], linewidth=1.8, marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"задержка {delay} с")
        ax.set_title(name, color=SIGNAL_COLOR[name], fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("комиссия, % за сторону", color=INK2, fontsize=9)
        ax.set_xticks([f * 100 for f in FEES])
        ax.set_xticklabels([f"{f * 100:g}" for f in FEES])
        if g.empty:
            ax.text(0.5, 0.5, "нет сделок", transform=ax.transAxes, ha="center", color=MUTED)
    axes[0].set_ylabel("средний чистый PnL на сделку, б.п.", color=INK2, fontsize=9)
    handles, _ = axes[-1].get_legend_handles_labels()
    if not handles:  # у последнего сигнала нет сделок — берём легенду с первой панели, где линии есть
        handles = next((a.get_legend_handles_labels()[0] for a in axes if a.get_legend_handles_labels()[0]), [])
    if handles:
        axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    finish(fig, f"Чувствительность к задержке и комиссии, горизонт {h} с", kind, path)


def plot_distribution(tr, h, delay, fee, kind, path):
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(3.3 * len(SIGNALS), 3.6), sharex=True)
    d = tr[(tr.h == h) & (tr.delay == delay)]
    net_all = (d.exec_ret - fee * d.fee_base) * BP
    both = pd.concat([net_all, d.gross_ret * BP])  # границы по чистому и валовому PnL вместе, чтобы не обрезать хвосты
    lo, hi = (np.percentile(both, [0.5, 99.5]) if len(both) else (-1, 1))
    bins = np.linspace(lo, hi, 40) if hi > lo else 20
    for ax, name in zip(axes, SIGNALS):
        style_ax(ax)
        g = d[d.signal == name]
        ax.set_title(name, color=SIGNAL_COLOR[name], fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("PnL на сделку, б.п.", color=INK2, fontsize=9)
        if g.empty:
            ax.text(0.5, 0.5, "нет сделок", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        net, gross = (g.exec_ret - fee * g.fee_base) * BP, g.gross_ret * BP
        ax.hist(net, bins=bins, color=SIGNAL_COLOR[name], edgecolor=SURFACE, linewidth=1, label="чистый")
        ax.hist(gross, bins=bins, histtype="step", color=INK2, linewidth=1.2, label="валовый")
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.text(0.98, 0.95, f"n={len(g)}\nсредний чистый {net.mean():.1f} б.п.", transform=ax.transAxes, ha="right",
                va="top", color=INK2, fontsize=8, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85))
    axes[0].set_ylabel("число сделок", color=INK2, fontsize=9)
    handles = next((a.get_legend_handles_labels() for a in axes if a.get_legend_handles_labels()[0]), None)
    if handles:
        axes[0].legend(*handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.8))
    finish(fig, f"Распределение PnL на сделку, горизонт {h} с, задержка {delay} с, комиссия {fee:.2%}/сторона", kind, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="синтетический рынок вместо data/predictions")
    p.add_argument("--delay", type=int, default=1, help="задержка исполнения для основных таблиц и графиков, с")
    p.add_argument("--fee", type=float, default=F.FEE, help="комиссия за сторону для основных таблиц и графиков")
    p.add_argument("--signal-fee", type=float, default=F.FEE, help="комиссия в пороге θ сигнала (зафиксирована заранее)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # вывод не должен падать при перенаправлении в cp1251

    if args.synthetic:
        market, preds = make_synthetic()
        kind, n_days = "synthetic", None
    else:
        if not (PRED_DIR / "_meta.json").exists():
            raise SystemExit("Нет прогнозов: сначала python train.py")
        market, preds, meta = load_real()
        kind, n_days = data_kind(meta), meta.get("n_days")
    print("\n" + banner(kind, n_days) + "\n")

    delays = sorted(set(DELAYS) | {args.delay})
    tr = run_backtest(market, preds, SIGNALS, delays, args.signal_fee)
    if tr.empty:
        print("Сделок нет ни у одного сигнала — нечего считать (при комиссии в пороге 0,05% это возможно).")
        return
    day_sets = {h: sorted({d for (tk, hh), p_ in preds.items() if hh == h for d in p_.day.unique()}) for h in {h for _, h in preds}}
    out = OUT_ROOT / kind
    out.mkdir(parents=True, exist_ok=True)
    pooled = summary_tables(tr, day_sets).assign(data_kind=kind)
    by_ticker = summary_tables(tr[tr.delay == args.delay], day_sets, group_by_ticker=True).assign(data_kind=kind)
    pooled.to_csv(out / "summary_pooled.csv", index=False)
    by_ticker.to_csv(out / "summary_by_ticker.csv", index=False)

    cols = ["signal", "h", "n_days", "n_trades", "avg_gross_bp", "avg_after_spread_bp", "avg_net_bp", "total_net_bp", "hit_gross",
            "hit_net", "sharpe_daily", "max_dd_bp", "breakeven_fee_pct"]
    head = pooled[(pooled.delay == args.delay) & np.isclose(pooled.fee_pct, args.fee * 100) & pooled.h.isin(MAIN_HORIZONS)]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"== Основная конфигурация: задержка {args.delay} с, комиссия {args.fee:.3%} за сторону (все тикеры вместе) ==")
    print(head[cols].round(3).to_string(index=False))
    print(f"\nSharpe по дням считается по {len(next(iter(day_sets.values())))} дням — при такой выборке это иллюстрация, не оценка.")

    for h in sorted(day_sets):
        if h in MAIN_HORIZONS:
            plot_cum_pnl(tr, h, args.delay, args.fee, kind, out / f"cum_pnl_h{h}.png")
            plot_sensitivity(pooled, h, kind, out / f"sensitivity_h{h}.png")
            plot_distribution(tr, h, args.delay, args.fee, kind, out / f"pnl_dist_h{h}.png")
    print(f"\nфайлы: {out}")
    print(banner(kind, n_days))


if __name__ == "__main__":
    main()
