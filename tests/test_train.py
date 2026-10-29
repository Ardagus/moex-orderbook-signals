"""Тесты train.py на синтетических данных: валидация без утечки, метрики на ручных примерах, поиск известного сигнала."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import features as F  # noqa: E402
import train  # noqa: E402


def make_data(n_days=4, rows=600, seed=0, planted=True):
    """Дни идут встык по времени (самый жёсткий случай для отступа). y_30 зависит от ofi_30, если planted."""
    rng = np.random.default_rng(seed)
    n = n_days * rows
    d = pd.DataFrame(rng.normal(size=(n, len(F.FEATURE_COLS))), columns=F.FEATURE_COLS)
    d["ticker"] = "X"
    d["ts"] = pd.date_range("2026-10-29 10:00", periods=n, freq="s", tz="UTC")
    d["day"] = [f"d{i // rows}" for i in range(n)]
    d["half_spread"] = 1e-4
    noise = rng.normal(0, 1e-4, n)
    d["y_30"] = (2e-4 * d.ofi_30 if planted else 0) + noise
    return d


def test_walk_forward_has_embargo_and_never_trains_on_future():
    d = make_data()
    folds = list(train.walk_forward(d))
    assert [day for day, _, _ in folds] == ["d1", "d2", "d3"]
    for day, tr, te in folds:
        assert tr.ts.max() <= te.ts.min() - pd.Timedelta(seconds=train.EMBARGO_S)  # отступ не меньше макс. горизонта
        assert tr.day.max() < day and (tr.ts < te.ts.min()).all()  # train строго раньше теста


def test_regression_finds_planted_signal():
    reg = train.run_regression(make_data(), 30)
    rows = pd.DataFrame(train.reg_table(reg, "X", 30)).set_index("model")
    assert rows.loc["ridge", "IC"] > 0.5 and rows.loc["ridge", "Spearman"] > 0.5
    assert rows.loc["ridge", "MAE_vs_zero"] < 1 and rows.loc["ridge", "R2"] > 0.2
    assert np.isnan(rows.loc["zero", "IC"]) and rows.loc["zero", "MAE_vs_zero"] == 1  # нулевой прогноз = эталон 1.0
    assert rows.loc["ridge", "folds_IC>0"] == "3/3"


def test_regression_on_pure_noise_has_no_signal():
    reg = train.run_regression(make_data(n_days=5, rows=800, seed=3, planted=False), 30)
    rows = pd.DataFrame(train.reg_table(reg, "X", 30)).set_index("model")
    assert abs(rows.loc["ridge", "IC"]) < 0.1 and rows.loc["ridge", "R2"] < 0.01


def test_classification_metrics_on_hand_example():
    preds = pd.DataFrame({"label": [0, 0, 1, -1], "majority_cls": 0, "imb1_cls": [1, 1, 1, -1],
                          "logit_cls": [0, 0, 0, 0], "lgbm_cls": [0, 0, 1, 0]})
    t = pd.DataFrame(train.cls_table(preds, "X", 30)).set_index("model")
    assert t.loc["majority", "acc"] == 0.5 and t.loc["majority", "acc_minus_base"] == 0
    assert t.loc["lgbm", "acc"] == 0.75 and t.loc["lgbm", "acc_minus_base"] == 0.25
    assert t.loc["lgbm", "rec_+1"] == 1 and t.loc["lgbm", "prec_+1"] == 1 and t.loc["lgbm", "rec_-1"] == 0
    assert t.loc["imb1", "acc"] == 0.5 and np.isclose(t.loc["imb1", "prec_+1"], 1 / 3)
    assert t.loc["lgbm", "share_-1"] == 0.25 and t.loc["lgbm", "share_0"] == 0.5


def test_majority_baseline_comes_from_train_not_test():
    d = make_data(planted=False)
    label = pd.Series(0.0, index=d.index)
    label[d.day == "d2"] = 1.0  # в тесте d2 почти все +1, а в train мажоритарный класс 0
    cls = train.run_classification(d, 30, label)
    fold = cls[cls.day == "d2"]
    assert (fold.majority_cls == 0).all()
    assert ((fold.label == fold.majority_cls).mean()) < 0.1  # бейзлайн на таком тесте честно плохой


def test_single_class_train_gives_constant_prediction():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pred = train.fit_predict_cls(train.cls_models()["logit"], x, pd.Series([0.0, 0.0, 0.0]), x)
    assert (pred == 0).all()
