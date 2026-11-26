# -*- coding: utf-8 -*-
"""rate_path のバックテスト。

過去の各月末を「その日」とみなして、その時点で見えていた利回りカーブから
rate_path.solve_path でパスを復元し、その後 1 年の実際の政策金利の動きと
比べる。出力は CSV（data/backtest_results.csv）で、集計は rate_path_eval.py。

注意:
  会合の日程は実際のカレンダー（後知恵）を使う。将来の会合日程が
  分からなかったことによる誤差は含まない。
  米国の実績は FF 金利誘導目標の上限（FRED: DFEDTARU）、日本の実績は
  無担保コール翌日物（日銀 FM01）で測る。

使い方:
  python3 rate_path_backtest.py [--us] [--jp]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from bisect import bisect_right
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rate_path as rp  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ 実際の会合日程（決定日）
FOMC_HIST = [
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
]
BOJ_HIST = [
    "2023-01-19", "2023-03-10", "2023-04-28", "2023-06-16", "2023-07-28",
    "2023-09-22", "2023-10-31", "2023-12-19",
    "2024-01-23", "2024-03-19", "2024-04-26", "2024-06-14", "2024-07-31",
    "2024-09-20", "2024-10-31", "2024-12-19",
    "2025-01-24", "2025-03-19", "2025-05-01", "2025-06-17", "2025-07-31",
    "2025-09-19", "2025-10-30", "2025-12-19",
    "2026-01-23", "2026-03-19", "2026-04-28", "2026-06-17", "2026-07-31",
    "2026-09-18", "2026-10-29", "2026-12-18",
]


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ------------------------------------------------------------ データ取得（キャッシュ付き）
def _cached(name: str, fetch) -> bytes:
    f = CACHE / name
    if f.exists():
        return f.read_bytes()
    raw = fetch()
    f.write_bytes(raw)
    return raw


def fred_series(series: str) -> list[tuple[date, float]]:
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
           "&cosd=2018-01-01&coed=2026-11-26")
    raw = _cached(f"fred_{series}.csv", lambda: rp._get(url, timeout=120))
    out = []
    for r in csv.reader(io.StringIO(raw.decode())):
        if len(r) == 2 and r[1] not in ("", "."):
            try:
                out.append((datetime.strptime(r[0], "%Y-%m-%d").date(), float(r[1])))
            except ValueError:
                pass
    return out


def mof_history() -> list[tuple[date, list[tuple[float, float]]]]:
    raw = _cached("jgbcm_all.csv", lambda: rp._get(
        "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv", timeout=120))
    out = []
    for r in csv.reader(io.StringIO(raw.decode("shift_jis"))):
        if r and r[0].strip() and (r[0].strip()[0] in "RH" or r[0].strip().startswith("S")):
            try:
                d = rp._parse_mof_date(r[0])
                out.append((d, [(1.0, float(r[1])), (2.0, float(r[2])), (3.0, float(r[3]))]))
            except (ValueError, IndexError):
                continue
    return sorted(out)


def boj_call_history() -> list[tuple[date, float]]:
    def fetch() -> bytes:
        url = ("https://www.stat-search.boj.or.jp/api/v1/getDataCode"
               "?db=FM01&code=STRDCLUCON&format=json&startdate=201801&enddate=202612")
        return rp._get(url, timeout=120)
    raw = _cached("boj_call.json", fetch)
    vals = json.loads(raw)["RESULTSET"][0]["VALUES"]
    out = []
    for dstr, v in zip(vals["SURVEY_DATES"], vals["VALUES"]):
        if v is not None:
            out.append((datetime.strptime(str(dstr), "%Y%m%d").date(), float(v)))
    return out


def jsda_obs_on(target: date) -> tuple[date, list[tuple[float, float]]] | tuple[None, None]:
    """target 以前で最も新しい JSDA 日次 CSV から短期の観測点を取る。"""
    for back in range(0, 8):
        d = target - timedelta(days=back)
        name = f"jsda_S{d.strftime('%y%m%d')}.csv"
        url = ("https://market.jsda.or.jp/shijyo/saiken/baibai/baisanchi/files/"
               f"{d.year}/S{d.strftime('%y%m%d')}.csv")
        try:
            raw = _cached(name, lambda u=url: rp._get(u, timeout=120))
        except Exception:
            continue
        rows = list(csv.reader(io.StringIO(raw.decode("shift_jis"))))
        data_date = datetime.strptime(rows[0][0], "%Y%m%d").date()
        points: list[tuple[float, float]] = []
        for r in rows:
            if len(r) < 15:
                continue
            nm = r[3]
            if "変動利付" in nm or "分離" in nm:
                continue
            if "国庫短期証券" not in nm and "国債" not in nm:
                continue
            try:
                mat = datetime.strptime(r[4], "%Y%m%d").date()
            except ValueError:
                continue
            T = (mat - data_date).days / 365.0
            if not (0.01 < T <= 1.5):
                continue
            yld = None
            for cell in (r[6], r[14]):
                try:
                    v = float(cell)
                except ValueError:
                    continue
                if 0 < v < 100:
                    yld = v
                    break
            if yld is not None:
                points.append((T, yld))
        obs = []
        for tgt in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
            if points:
                obs.append(min(points, key=lambda pt: abs(pt[0] - tgt)))
        return data_date, sorted(set(obs))
    return None, None


# ------------------------------------------------------------ 時系列の参照
class Series:
    def __init__(self, pts: list[tuple[date, float]]):
        self.dates = [p[0] for p in pts]
        self.vals = [p[1] for p in pts]

    def at(self, d: date) -> float | None:
        i = bisect_right(self.dates, d) - 1
        return self.vals[i] if i >= 0 else None


def month_ends(start: date, end: date) -> list[date]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        out.append(nxt - timedelta(days=1))
        y, m = nxt.year, nxt.month
    return [d for d in out if start <= d <= end]


# ------------------------------------------------------------ バックテスト本体
def run_us(eval_dates: list[date]) -> list[dict]:
    tenors = [(1 / 12, "DGS1MO"), (3 / 12, "DGS3MO"), (6 / 12, "DGS6MO"),
              (1.0, "DGS1"), (2.0, "DGS2"), (3.0, "DGS3")]
    curves = {sid: Series(fred_series(sid)) for _, sid in tenors}
    sofr = Series(fred_series("SOFR"))
    target = Series(fred_series("DFEDTARU"))
    cal = [_d(s) for s in FOMC_HIST]
    rows = []
    for e in eval_dates:
        obs, ok = [], True
        for T, sid in tenors:
            v = curves[sid].at(e)
            if v is None:
                ok = False
                break
            obs.append((T, v))
        s = sofr.at(e)
        if not ok or s is None:
            continue
        r0 = s * 365.0 / 360.0
        meetings = rp.meeting_dates(FOMC_HIST, e)
        res = rp.solve_path(r0, obs, meetings, e, 100.0)
        rows.append(_score("US", e, res, target, cal))
    return rows


def run_jp(eval_dates: list[date]) -> list[dict]:
    mof = mof_history()
    mof_dates = [p[0] for p in mof]
    call = Series(boj_call_history())
    cal = [_d(s) for s in BOJ_HIST]
    rows = []
    for e in eval_dates:
        i = bisect_right(mof_dates, e) - 1
        if i < 0:
            continue
        _, obs = jsda_obs_on(e)
        if obs is None:
            continue
        r0 = call.at(e)
        if r0 is None:
            continue
        meetings = rp.meeting_dates(BOJ_HIST, e)
        res = rp.solve_path(r0, sorted(mof[i][1] + obs), meetings, e, 6.25)
        rows.append(_score("JP", e, res, call, cal))
    return rows


def _score(country: str, e: date, res: dict, policy: Series, cal: list[date]) -> dict:
    """復元したパスと、その後 1 年の実績を並べた 1 行を作る。"""
    lvl0 = policy.at(e)
    lvl1 = policy.at(e + timedelta(days=365))
    realized_moves = (lvl1 - lvl0) / rp.STEP if lvl0 is not None and lvl1 is not None else None
    # 実績の最初の 1 回分（0.25%）の動き: 会合ごとに水準を見て最初に届いた会合。
    # 政策金利の系列は決定の翌営業日から新しい水準になるので、会合の 7 日後で測る。
    direction = res["direction"]
    first_move = None
    if direction != 0 and lvl0 is not None:
        for m in cal:
            if e < m <= e + timedelta(days=365):
                lv = policy.at(m + timedelta(days=7))
                if lv is not None and direction * (lv - lvl0) >= rp.STEP - 1e-9:
                    first_move = m
                    break
    # 次の会合が実際に動いたか（p1 の答え合わせ用）
    nxt = res["meetings"][0]
    lv_next = policy.at(nxt + timedelta(days=7))
    next_moved = None
    if lvl0 is not None and lv_next is not None:
        next_moved = 1 if direction != 0 and direction * (lv_next - lvl0) >= rp.STEP - 1e-9 else 0
    return {
        "country": country, "eval_date": e.isoformat(), "r0": round(res["r0"], 3),
        "implied_n1y": round(res["n_1y"], 2),
        "implied_half": res["half"].isoformat() if res["half"] else "",
        "implied_full": res["full"].isoformat() if res["full"] else "",
        "implied_p1": round(res["p"][0], 2),
        "realized_moves_1y": round(realized_moves, 2) if realized_moves is not None else "",
        "realized_first_move": first_move.isoformat() if first_move else "",
        "next_meeting": nxt.isoformat(), "next_moved": next_moved if next_moved is not None else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", action="store_true")
    ap.add_argument("--jp", action="store_true")
    args = ap.parse_args()
    both = not (args.us or args.jp)
    rows: list[dict] = []
    if both or args.us:
        rows += run_us(month_ends(date(2019, 1, 1), date(2025, 10, 31)))
        print(f"US done: {len(rows)} evals", file=sys.stderr)
    if both or args.jp:
        jp = run_jp(month_ends(date(2024, 1, 1), date(2025, 10, 31)))
        rows += jp
        print(f"JP done: {len(jp)} evals", file=sys.stderr)
    out = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
