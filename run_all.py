"""Весь пайплайн одной командой (без сбора данных: collect.py запускается отдельно, нужен токен и время).

    python run_all.py                    # реальные данные: prepare -> features -> train -> backtest -> тесты
    python run_all.py --pipeline-check   # проверка пайплайна: псевдо-дни + синтетика (результаты помечены SYNTHETIC / PIPELINE CHECK)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def step(title, *args):
    print(f"\n=== {title} ===", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1"}  # иначе на Windows вывод со знаком θ падает в кодировке cp1251
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env=env)  # check=True: остановиться на первой ошибке


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-check", action="store_true", help="псевдо-дни и синтетика вместо реальной валидации по дням")
    p.add_argument("--session", default=None, choices=["main", "evening", "all"],
                   help="сессия для обучения (по умолчанию main; для --pipeline-check evening)")
    p.add_argument("--block-minutes", type=float, default=3, help="длина псевдо-дня для --pipeline-check")
    args = p.parse_args()

    if not (ROOT / "tinvest_pb" / "marketdata_pb2.py").exists():
        step("0. Генерация gRPC-клиента из contracts/", "gen_proto.py")
    step("1. Подготовка данных (сетка 1 с)", "prepare.py")
    step("2. Признаки и цели", "features.py")
    if args.pipeline_check:
        step("3. Обучение на псевдо-днях (ПРОВЕРКА ПАЙПЛАЙНА)", "train.py", "--session", args.session or "evening",
             "--block-minutes", str(args.block_minutes))
        step("4a. Бэктест на псевдо-днях (ПРОВЕРКА ПАЙПЛАЙНА)", "backtest.py")
        step("4b. Бэктест на синтетическом рынке (SYNTHETIC)", "backtest.py", "--synthetic")
    else:
        step("3. Обучение, walk-forward по дням", "train.py", "--session", args.session or "main")
        step("4. Бэктест", "backtest.py")
    step("5. Тесты", "-m", "pytest", "-q")
    print("\nГотово. Таблицы: results/, графики и сводки бэктеста: results/backtest/<вид данных>/")


if __name__ == "__main__":
    main()
