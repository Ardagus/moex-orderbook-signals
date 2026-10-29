"""Этап 4: модели и walk-forward валидация по дням (train дни 1..k, test день k+1, без перемешивания).

Две постановки (признаки одни и те же, mid в признаки не входит):
  * регрессия y_h — оценка самого сигнала: IC (Пирсон), Spearman, MAE против нулевого прогноза, R²;
  * 3 класса (-1 / 0 / +1) с порогом θ = 2·(половина спреда + комиссия) — основа для торговли и бэктеста.
Гиперпараметры зафиксированы в коде и на тестовых днях не подбираются.

Запуск: python train.py [--session main] [--fee 0.0005] [--block-minutes N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_recall_fscore_support, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "data" / "predictions"
RESULTS_DIR = ROOT / "results"
MAIN_HORIZONS = [30, 60]  # основные; 5 с — дополнительный
EXTRA_HORIZONS = [5]
EMBARGO_S = max(F.HORIZONS)  # отступ между train и test не меньше максимального горизонта
MIN_TRAIN_ROWS = 200
Y_SCALE = 1e4  # доходности в б.п. (1e-4): так деревьям и MAE удобнее
FEES = [0.0, 0.0002, 0.0005, 0.001]  # сетка для чувствительности к комиссии (доля от оборота за сторону)
CLASSES = [-1, 0, 1]
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def load_features(session, block_minutes=None):
    df = pd.concat([pd.read_parquet(f) for f in sorted(F.OUT_DIR.glob("*/*.parquet"))], ignore_index=True)
    if session != "all":
        df = df[df.session == session]
    df = df.sort_values(["ticker", "ts"]).reset_index(drop=True)
    df["day"] = (df.ts + pd.Timedelta(hours=3)).dt.strftime("%Y-%m-%d")  # торговый день по МСК
    if block_minutes:  # только для проверки пайплайна на малых данных: режем день на «псевдо-дни»
        start = df.groupby(["ticker", "day"]).ts.transform("min")
        block = ((df.ts - start).dt.total_seconds() // (block_minutes * 60)).astype(int)
        df["day"] = df.day + "_b" + block.map("{:02d}".format)
    return df


def walk_forward(d, min_train_days=1):
    """Для каждого дня k+1: train = дни 1..k без последних EMBARGO_S секунд перед тестом, test = день k+1."""
    days = sorted(d.day.unique())
    for k in range(min_train_days, len(days)):
        test = d[d.day == days[k]]
        cutoff = test.ts.min() - pd.Timedelta(seconds=EMBARGO_S)  # цели train-строк не должны доставать до теста
        train = d[d.day.isin(days[:k]) & (d.ts <= cutoff)]
        if len(train) >= MIN_TRAIN_ROWS and len(test) > 0:
            yield days[k], train, test


def reg_models():
    return {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "lgbm": LGBMRegressor(**LGB_PARAMS)}


def cls_models():
    return {"logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
            "lgbm": LGBMClassifier(**LGB_PARAMS)}


def run_regression(d, h):
    """Прогноз y_h на тестовых днях. Возвращает таблицу прогнозов (в долях, не в б.п.)."""
    d = d.dropna(subset=F.FEATURE_COLS + [f"y_{h}"])
    parts = []
    for day, train, test in walk_forward(d):
        out = test[["ticker", "ts", "day", f"y_{h}"]].rename(columns={f"y_{h}": "y"})
        for name, model in reg_models().items():
            model.fit(train[F.FEATURE_COLS], train[f"y_{h}"] * Y_SCALE)
            out[f"{name}_y"] = model.predict(test[F.FEATURE_COLS]) / Y_SCALE
        out["zero_y"] = 0.0  # бейзлайн «ничего не предсказываем»
        out["imb1_y"] = test.imb_1.to_numpy()  # бейзлайн-правило: знак/величина imbalance_1 (масштаб не важен для IC)
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fit_predict_cls(model, x_train, y_train, x_test):
    if y_train.nunique() < 2:  # в train один класс — модели нечему учиться, прогноз константный
        return np.full(len(x_test), y_train.iloc[0])
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run_classification(d, h, label):
    """label — Series с классами (индекс как у d). Возвращает прогнозы классов + бейзлайны."""
    d = d.assign(label=label).dropna(subset=F.FEATURE_COLS + ["label"])
    parts = []
    for day, train, test in walk_forward(d):
        out = test[["ticker", "ts", "day", "label"]].copy()
        for name, model in cls_models().items():
            out[f"{name}_cls"] = fit_predict_cls(model, train[F.FEATURE_COLS], train.label, test[F.FEATURE_COLS])
        out["majority_cls"] = train.label.mode()[0]  # «всегда самый частый класс» — по обучающей выборке
        out["imb1_cls"] = np.sign(test.imb_1.to_numpy())  # правило: знак imbalance_1
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------- метрики ----------

def ic(a, b, spearman=False):
    if np.std(b) == 0 or np.std(a) == 0:
        return np.nan  # константный прогноз корреляции не имеет
    return spearmanr(a, b)[0] if spearman else np.corrcoef(a, b)[0, 1]


def reg_table(preds, ticker, h):
    rows = []
    y = preds.y.to_numpy()
    mae_zero = np.abs(y).mean() * Y_SCALE
    for name in ("zero", "imb1", "ridge", "lgbm"):
        p = preds[f"{name}_y"].to_numpy()
        fold_ic = [ic(g[f"{name}_y"], g.y) for _, g in preds.groupby("day")]
        has_scale = name != "imb1"  # у imbalance_1 нет масштаба доходности, MAE и R² не имеют смысла
        mae = np.abs(y - p).mean() * Y_SCALE if has_scale else np.nan
        rows.append({"ticker": ticker, "h": h, "model": name, "n": len(y), "IC": ic(p, y), "Spearman": ic(p, y, True),
                     "MAE_bp": mae, "MAE_vs_zero": mae / mae_zero if has_scale else np.nan,
                     "R2": r2_score(y, p) if has_scale else np.nan,
                     "folds_IC>0": f"{int(np.nansum(np.array(fold_ic) > 0))}/{len(fold_ic)}"})
    return rows


def cls_table(preds, ticker, h, extra=None):
    rows = []
    y = preds.label.to_numpy()
    for name in ("majority", "imb1", "logit", "lgbm"):
        p = preds[f"{name}_cls"].to_numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
        base = (y == preds.majority_cls.to_numpy()).mean()  # точность бейзлайна «всегда самый частый класс»
        row = {"ticker": ticker, "h": h, "model": name, "n": len(y), "acc": (p == y).mean(), "acc_majority": base,
               "acc_minus_base": (p == y).mean() - base}
        for i, c in enumerate(CLASSES):
            n = "0" if c == 0 else f"{c:+d}"  # имена колонок: -1, 0, +1
            row.update({f"share_{n}": (y == c).mean(), f"prec_{n}": prec[i], f"rec_{n}": rec[i], f"f1_{n}": f1[i]})
        rows.append({**row, **(extra or {})})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="main", choices=["main", "evening", "all"])
    p.add_argument("--fee", type=float, default=F.FEE, help="комиссия за одну сторону, доля от оборота")
    p.add_argument("--block-minutes", type=float, help="ТОЛЬКО для проверки пайплайна: резать данные на псевдо-дни")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # вывод со знаком θ не должен падать при перенаправлении в cp1251
    kind = "pipeline_check_pseudo_days" if args.block_minutes else "real"
    out_dir = RESULTS_DIR / kind  # результаты проверки пайплайна лежат отдельно и не смешиваются с реальными
    out_dir.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features(args.session, args.block_minutes)
    if args.block_minutes:
        print("!!! ПРОВЕРКА ПАЙПЛАЙНА на псевдо-днях по", args.block_minutes, "мин — это НЕ результат исследования !!!\n")
    print(f"данные: session={args.session}, дней (или псевдо-дней): {df.day.nunique()}, строк: {len(df)}, "
          f"θ = 2·(half_spread + {args.fee:.4%})")
    for old in PRED_DIR.glob("*.parquet"):  # прогнозы прошлого запуска не должны смешиваться с новыми
        old.unlink()
    # метка происхождения прогнозов: по ней backtest.py решает, реальные это данные или проверка пайплайна
    (PRED_DIR / "_meta.json").write_text(json.dumps({
        "block_minutes": args.block_minutes, "fee": args.fee, "session": args.session,
        "n_days": int(df.day.nunique())}), encoding="utf-8")
    reg_rows, cls_rows, diag_rows, fee_rows = [], [], [], []
    for ticker, d in df.groupby("ticker"):
        for h in MAIN_HORIZONS + EXTRA_HORIZONS:
            reg = run_regression(d, h)
            if reg.empty:
                print(f"{ticker} h={h}: недостаточно дней для walk-forward, пропуск")
                continue
            reg_rows += reg_table(reg, ticker, h)
            y = d[f"y_{h}"]
            cls = run_classification(d, h, F.make_labels(y, d.half_spread, args.fee))
            cls_rows += cls_table(cls, ticker, h)
            merged = reg.merge(cls.drop(columns=["label"]), on=["ticker", "ts", "day"], how="left")
            merged.to_parquet(PRED_DIR / f"{ticker}_h{h}.parquet", index=False)  # прогнозы для бэктеста
            # диагностика: только строки, где mid изменился; НЕ используется в бэктесте
            moved = d[y.notna() & (y != 0)]
            diag = run_classification(moved, h, np.sign(moved[f"y_{h}"]))
            if not diag.empty:
                diag_rows += cls_table(diag, ticker, h)
            # чувствительность классификации к комиссии
            for fee in FEES:
                sens = run_classification(d, h, F.make_labels(y, d.half_spread, fee))
                fee_rows += cls_table(sens, ticker, h, {"fee": fee})

    tables = {"metrics_reg": reg_rows, "metrics_cls": cls_rows, "metrics_diag_nonzero": diag_rows,
              "metrics_fee_sensitivity": fee_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).assign(data_kind=kind).to_csv(out_dir / f"{name}.csv", index=False)
    if not reg_rows:
        print("Нет ни одного фолда: нужно минимум 2 торговых дня (или --block-minutes для проверки пайплайна).")
        return
    (out_dir / "run_config.json").write_text(json.dumps({
        "session": args.session, "fee": args.fee, "theta": "2 * (half_spread(t) + fee)", "embargo_s": EMBARGO_S,
        "lgbm_params": LGB_PARAMS, "ridge_alpha": 1.0, "features": F.FEATURE_COLS,
        "smoke_test_block_minutes": args.block_minutes}, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    reg_df, cls_df = pd.DataFrame(reg_rows), pd.DataFrame(cls_rows)
    print("\n== Регрессия y_h (IC/Spearman — сигнал; MAE_vs_zero < 1 — лучше нулевого прогноза) ==")
    print(reg_df.round(4).to_string(index=False))
    print("\n== 3 класса, θ = 2·(half_spread + комиссия) (acc_minus_base > 0 — лучше «всегда самого частого») ==")
    print(cls_df[["ticker", "h", "model", "n", "acc", "acc_majority", "acc_minus_base", "share_-1", "share_+1",
                  "rec_-1", "rec_+1", "prec_-1", "prec_+1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
