# -*- coding: utf-8 -*-
"""rate_path_backtest.py の結果 CSV を集計して精度を出す。

見るもの:
  1. 1 年後までの累計回数 N(1y): 平均誤差（バイアス）・MAE・RMSE・相関
  2. 方向の的中: 実績が 1 回以上動いた局面で、符号が合っていた割合
  3. 「半分以上見込む会合」と実績の最初の 1 回分の動きのタイミング差
  4. 次の会合の p1 と、実際に次の会合が動いたかの照合（較正）

使い方:
  python3 rate_path_eval.py [data/backtest_results.csv]
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _d(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date() if s else None


def report(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"== {label}: データなし")
        return
    imp = np.array([r["implied_n1y"] for r in rows])
    real = np.array([r["realized_moves_1y"] for r in rows])
    err = imp - real
    print(f"== {label}（n={len(rows)}、{rows[0]['eval_date']} 〜 {rows[-1]['eval_date']}）")
    print(f"  N(1y) バイアス {err.mean():+.2f} 回 / MAE {np.abs(err).mean():.2f} 回 / "
          f"RMSE {np.sqrt((err ** 2).mean()):.2f} 回 / 相関 {np.corrcoef(imp, real)[0, 1]:.2f}")

    moved = np.abs(real) >= 1.0
    if moved.sum():
        hit = (np.sign(imp[moved]) == np.sign(real[moved])).mean()
        print(f"  方向の的中（実績が 1 回以上動いた {int(moved.sum())} 局面）: {hit:.0%}")
    called = np.abs(imp) >= 1.0
    if called.sum():
        hit2 = (np.sign(real[called]) == np.sign(imp[called])).mean()
        print(f"  方向の的中（予想が 1 回以上だった {int(called.sum())} 局面）: {hit2:.0%}")

    # 半分超え会合 vs 実績の最初の動き（両方ある場合だけ）
    diffs = []
    for r in rows:
        h, f = _d(r["implied_half"]), _d(r["realized_first_move"])
        if h and f:
            diffs.append((h - f).days)
    if diffs:
        diffs = np.array(diffs)
        print(f"  半分超え会合のタイミング（n={len(diffs)}）: 中央値 {np.median(diffs):+.0f} 日 "
              f"（負 = 予想が早い）/ ±45 日以内 {(np.abs(diffs) <= 45).mean():.0%}")

    # p1 の較正: 次の会合が実際にその方向へ 1 回分動いたか
    buckets = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    print("  次の会合の p1 と実際の動き:")
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= min(max(r["implied_p1"], 0.0), 1.0) < hi
               and r["next_moved"] != ""]
        if sel:
            freq = np.mean([r["next_moved"] for r in sel])
            print(f"    p1 が {lo:.0%}〜{hi:.0%} の {len(sel)} 件: 実際に動いた {freq:.0%}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["implied_n1y"] = float(r["implied_n1y"])
        r["realized_moves_1y"] = float(r["realized_moves_1y"])
        r["implied_p1"] = float(r["implied_p1"])
        r["next_moved"] = int(r["next_moved"]) if r["next_moved"] != "" else ""
    report([r for r in rows if r["country"] == "US"], "米国")
    report([r for r in rows if r["country"] == "JP"], "日本")


if __name__ == "__main__":
    main()
