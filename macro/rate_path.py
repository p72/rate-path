# -*- coding: utf-8 -*-
"""市場金利から利上げ・利下げの時期を出す計算。

考え方:
  国債の利回り y(T) は、満期までの翌日物金利（政策金利にほぼ等しい）の
  平均の見込みに近似できる。政策金利は会合の翌日にしか変わらないので、
  見込みのパスは会合ごとの階段関数になる。各年限の利回りに最も合う階段を
  罰則付き最小二乗で解き、会合ごとの段差から「その会合で 0.25% 動く確率の
  目安」と累計の回数を読み取る（FedWatch と同じ考え方）。

    y(T) ≒ (1/T) [ r0*min(T, t1) + Σ_k x_k * l_k(T) ]
    l_k(T) = max(0, min(T, t_{k+1}) - t_k)

  最小化するのは
    Σ_j (W w_j)^2 (y_j - ŷ(T_j))^2 + λ^2 [ (x1 - r0)^2 + Σ_{k>=2} (x_k - x_{k-1})^2 ]
  前半が利回りへの当てはめ、後半が「隣の会合との差」を小さくする平滑化で、
  年限が足りない遠い先をなめらかに埋める。w_j = min(1, T_j / 0.5) は
  満期の近い国債ほど小さくする重み（小さな値動きで利回りが大きく飛ぶため）。

注意:
  国債の利回りには期間プレミアムや需給のゆがみが乗るので、回数・確率は目安。
  金利先物・OIS から出す市場の織り込み（FedWatch など）とはずれる。

入力データ:
  米国: FRED の国債金利（DGS: 1・3・6 か月、1・2・3 年。米財務省の日次
        パーイールドと同系統の H.15 ベース）。r0 = SOFR * 365/360
        （短期国債はレポ金利に近いので SOFR を基準にする）。
  日本: 財務省の国債金利（1・2・3 年）＋ 日本証券業協会の売買参考統計値
        （国庫短期証券と残り 1.5 年以内の国債）。r0 = 無担保コール翌日物
        （日銀時系列統計 FM01 / STRDCLUCON）。
  会合の日程: FRB と日本銀行の公表分を内蔵。知られていない先の会合は
        過去の間隔（中央値）で補う。

使い方:
  python3 rate_path.py            # 日米両方
  python3 rate_path.py --jp       # 日本だけ
  python3 rate_path.py --us --verbose   # 会合ごとの表も出す
  python3 rate_path.py --json     # 機械可読な出力
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta

import numpy as np

W_BIG = 100.0          # 利回りへの当てはめの重み（仕様: W = 100）
STEP = 0.25            # 1 回の利上げ/利下げ幅 (%)
HORIZON_YEARS = 3.0    # 何年先の会合まで解くか
UA = {"User-Agent": "rate-path/1.0 (personal research)"}

# ---------------------------------------------------------------- 会合日程
# 決定日（会合 2 日目）。公表済みの分だけを入れ、先は間隔で補う。
FOMC_DATES = [
    "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]
BOJ_DATES = [
    "2026-12-18",
    "2027-01-22", "2027-03-18", "2027-04-28", "2027-06-17",
    "2027-07-31", "2027-09-19", "2027-10-29", "2027-12-18",
]


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def meeting_dates(known: list[str], asof: date, horizon_years: float = HORIZON_YEARS) -> list[date]:
    """asof より後、ホライズンまでの会合日。不足分は過去間隔の中央値で補う。"""
    ds = sorted(_d(s) for s in known)
    ds = [d for d in ds if d > asof]
    if len(ds) >= 2:
        gaps = [(ds[i + 1] - ds[i]).days for i in range(len(ds) - 1)]
        step = int(np.median(gaps))
    else:
        step = 45
    end = asof + timedelta(days=int(horizon_years * 365))
    while ds and ds[-1] < end:
        ds.append(ds[-1] + timedelta(days=step))
    return [d for d in ds if d <= end]


# ---------------------------------------------------------------- ソルバー
def solve_path(r0: float, obs: list[tuple[float, float]], meetings: list[date],
               asof: date, lam: float, w_big: float = W_BIG) -> dict:
    """obs = [(残存年数 T, 利回り %)]。会合ごとの階段 x_k を解く。"""
    t = np.array([(m - asof).days / 365.0 for m in meetings])   # 会合までの年数
    K = len(t)
    T = np.array([o[0] for o in obs])
    y = np.array([o[1] for o in obs])

    # l_k(T): 満期までに x_k が続く年数。最後の段は満期まで続く (t_{K+1} = ∞)
    t_ext = np.concatenate([t, [np.inf]])
    L = np.maximum(0.0, np.minimum(T[:, None], t_ext[None, 1:]) - t_ext[None, :-1])  # (J, K)
    base = r0 * np.minimum(T, t[0]) / T                       # r0 が担う部分
    A = L / T[:, None]
    w = np.minimum(1.0, T / 0.5) * w_big
    A_w = A * w[:, None]
    b_w = (y - base) * w

    # 平滑化の罰則行: λ(x1 - r0), λ(x_k - x_{k-1})
    P = np.zeros((K, K))
    for k in range(K):
        P[k, k] = lam
        if k >= 1:
            P[k, k - 1] = -lam
    rhs_p = np.zeros(K)
    rhs_p[0] = lam * r0

    M = np.vstack([A_w, P])
    rhs = np.concatenate([b_w, rhs_p])
    x, *_ = np.linalg.lstsq(M, rhs, rcond=None)

    x_prev = np.concatenate([[r0], x[:-1]])
    p = (x - x_prev) / STEP            # その会合で動く回数の目安（確率 ≒ p）
    N = (x - r0) / STEP                # その会合までの累計回数

    def level_at(years: float) -> float:
        """asof から years 年後の時点で効いている階段の水準。"""
        k = int(np.searchsorted(t, years, side="right")) - 1
        return r0 if k < 0 else float(x[k])

    n_1y = (level_at(1.0) - r0) / STEP
    direction = 0 if abs(n_1y) < 1e-9 else (1 if n_1y > 0 else -1)
    a = direction * N                  # 方向を正にした累計回数

    def first_cross(thr: float) -> int | None:
        idx = np.where(a >= thr)[0]
        return int(idx[0]) if len(idx) else None

    half_i = first_cross(0.5)
    full_i = first_cross(1.0)
    ambiguous = False
    if half_i is not None and direction != 0:
        prev_a = a[half_i - 1] if half_i >= 1 else 0.0
        if (0.3 <= prev_a < 0.5) or a[half_i] < 0.7:
            ambiguous = True           # 2 つの会合で割れている

    fitted = base + A @ x
    return {
        "r0": r0, "meetings": meetings, "x": x.tolist(), "p": p.tolist(),
        "N": N.tolist(), "n_1y": float(n_1y), "direction": direction,
        "half": meetings[half_i] if half_i is not None else None,
        "full": meetings[full_i] if full_i is not None else None,
        "ambiguous": ambiguous,
        "fit_rmse_bp": float(np.sqrt(np.mean((y - fitted) ** 2)) * 100),
    }


# ---------------------------------------------------------------- 取得まわり
def _get(url: str, timeout: int = 60) -> bytes:
    # この環境では FRED 等が Python 系クライアントの UA を受け付けず
    # 応答が止まる（curl の UA では即応答）ため、取得は curl に寄せる。
    import subprocess
    p = subprocess.run(
        ["curl", "-sS", "--fail", "--compressed", "--max-time", str(timeout), url],
        capture_output=True, timeout=timeout + 10)
    if p.returncode != 0:
        raise RuntimeError(f"curl failed ({p.returncode}): {p.stderr.decode()[:200]}")
    return p.stdout


def _fred_latest(series: str) -> tuple[date, float]:
    # 全期間だと CSV が大きくて遅いので直近分だけ取る
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
           "&cosd=2026-01-01")
    txt = _get(url, timeout=120).decode()
    rows = [r for r in csv.reader(io.StringIO(txt)) if len(r) == 2 and r[1] not in ("", ".")]
    dstr, val = rows[-1]
    return datetime.strptime(dstr, "%Y-%m-%d").date(), float(val)


def fetch_us() -> tuple[date, float, list[tuple[float, float]]]:
    """(データ日, r0, [(T, 利回り)]) を返す。"""
    tenors = [(1 / 12, "DGS1MO"), (3 / 12, "DGS3MO"), (6 / 12, "DGS6MO"),
              (1.0, "DGS1"), (2.0, "DGS2"), (3.0, "DGS3")]
    obs, asof = [], None
    for T, sid in tenors:
        d, v = _fred_latest(sid)
        obs.append((T, v))
        asof = d if asof is None or d > asof else asof
    _, sofr = _fred_latest("SOFR")
    return asof, sofr * 365.0 / 360.0, obs


def _parse_mof_date(s: str) -> date:
    m = re.match(r"R(\d+)\.(\d+)\.(\d+)", s.strip())
    if m:  # 令和
        return date(2018 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"H(\d+)\.(\d+)\.(\d+)", s.strip())
    if m:  # 平成
        return date(1988 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise ValueError(f"unknown MOF date: {s}")


def fetch_jp_mof() -> tuple[date, list[tuple[float, float]]]:
    txt = _get("https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv").decode("shift_jis")
    rows = [r for r in csv.reader(io.StringIO(txt)) if r and re.match(r"[RH]\d", r[0])]
    header_note = rows[-1]
    d = _parse_mof_date(header_note[0])
    # 列: 基準日, 1年, 2年, 3年, 4年, 5年, ...
    return d, [(1.0, float(header_note[1])), (2.0, float(header_note[2])), (3.0, float(header_note[3]))]


def fetch_jp_jsda(asof_hint: date) -> tuple[date, list[tuple[float, float]]]:
    """JSDA 売買参考統計値から、国庫短期証券・残り 1.5 年以内の国債の利回りを拾う。

    CSV 列（実測）: [3]=銘柄名, [4]=償還期日, [6]=平均値・複利利回り,
    [14]=平均値・単利利回り（国庫短期証券はこちら）。目標年限に最も近い
    銘柄を代表点にする。
    """
    raw, used = None, None
    for back in range(0, 11):
        d = asof_hint - timedelta(days=back)
        url = ("https://market.jsda.or.jp/shijyo/saiken/baibai/baisanchi/files/"
               f"{d.year}/S{d.strftime('%y%m%d')}.csv")
        try:
            raw = _get(url)
            used = d
            break
        except Exception:
            continue
    if raw is None:
        raise RuntimeError("JSDA の売買参考統計値を取得できなかった")
    rows = list(csv.reader(io.StringIO(raw.decode("shift_jis"))))
    data_date = datetime.strptime(rows[0][0], "%Y%m%d").date()
    points: list[tuple[float, float]] = []
    for r in rows:
        if len(r) < 15:
            continue
        name = r[3]
        if "変動利付" in name or "分離" in name:
            continue
        if "国庫短期証券" not in name and "国債" not in name:
            continue
        try:
            mat = datetime.strptime(r[4], "%Y%m%d").date()
        except ValueError:
            continue
        T = (mat - data_date).days / 365.0
        if not (0.01 < T <= 1.5):
            continue
        yld = None
        for cell in (r[6], r[14]):     # 複利 → 単利の順に有効な値を探す
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
    for target in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        if points:
            obs.append(min(points, key=lambda pt: abs(pt[0] - target)))
    # 重複を除く
    obs = sorted(set(obs))
    return used, obs


def fetch_jp_r0() -> tuple[date, float]:
    """日銀時系列統計 FM01 / STRDCLUCON（無担保コール O/N 平均）の最新値。"""
    now = date.today()
    start = (now - timedelta(days=200)).strftime("%Y%m")
    end = now.strftime("%Y%m")
    url = ("https://www.stat-search.boj.or.jp/api/v1/getDataCode"
           f"?db=FM01&code=STRDCLUCON&format=json&startdate={start}&enddate={end}")
    d = json.loads(_get(url))
    vals = d["RESULTSET"][0]["VALUES"]
    dates, series = vals["SURVEY_DATES"], vals["VALUES"]
    for i in range(len(dates) - 1, -1, -1):
        if series[i] is not None:
            return datetime.strptime(str(dates[i]), "%Y%m%d").date(), float(series[i])
    raise RuntimeError("コールレートの値が取れなかった")


def fetch_jp() -> tuple[date, float, list[tuple[float, float]]]:
    mof_date, obs = fetch_jp_mof()
    _, short_obs = fetch_jp_jsda(mof_date)
    _, r0 = fetch_jp_r0()
    return mof_date, r0, sorted(obs + short_obs)


# ---------------------------------------------------------------- 表示
def summarize(label: str, asof: date, res: dict) -> str:
    arrow = "+" if res["n_1y"] >= 0 else ""
    nxt = res["meetings"][0]
    p1 = res["p"][0] * 100
    half = res["half"].isoformat() if res["half"] else "―"
    full = res["full"].isoformat() if res["full"] else "―"
    amb = "（半分超えは2会合で割れている）" if res["ambiguous"] else ""
    return (f"{label}（{asof.isoformat()}）：r₀ {res['r0']:.2f}%　"
            f"次の会合 {nxt.isoformat()} の p = {p1:.0f}%　"
            f"半分 {half}　見込みきる {full}　"
            f"1年後までに {arrow}{res['n_1y']:.1f} 回{amb}")


def detail(res: dict) -> str:
    lines = ["  会合日        金利 x_k    p_k     累計 N_k"]
    for m, x, p, n in zip(res["meetings"], res["x"], res["p"], res["N"]):
        lines.append(f"  {m.isoformat()}  {x:7.3f}%  {p * 100:5.0f}%   {n:+.2f}")
    lines.append(f"  利回りへの当てはめ誤差: {res['fit_rmse_bp']:.1f} bp")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="市場金利から利上げ・利下げの時期を出す")
    ap.add_argument("--us", action="store_true", help="米国だけ")
    ap.add_argument("--jp", action="store_true", help="日本だけ")
    ap.add_argument("--verbose", action="store_true", help="会合ごとの表も出す")
    ap.add_argument("--json", action="store_true", help="JSON で出す")
    ap.add_argument("--lam-us", type=float, default=100.0)
    ap.add_argument("--lam-jp", type=float, default=6.25)
    args = ap.parse_args()
    both = not (args.us or args.jp)
    out = {}

    if both or args.us:
        asof, r0, obs = fetch_us()
        res = solve_path(r0, obs, meeting_dates(FOMC_DATES, asof), asof, args.lam_us)
        out["US"] = {"asof": asof.isoformat(), **res,
                     "meetings": [m.isoformat() for m in res["meetings"]],
                     "half": res["half"].isoformat() if res["half"] else None,
                     "full": res["full"].isoformat() if res["full"] else None}
        if not args.json:
            print(summarize("米国", asof, res))
            if args.verbose:
                print(detail(res))
    if both or args.jp:
        asof, r0, obs = fetch_jp()
        res = solve_path(r0, obs, meeting_dates(BOJ_DATES, asof), asof, args.lam_jp)
        out["JP"] = {"asof": asof.isoformat(), **res,
                     "meetings": [m.isoformat() for m in res["meetings"]],
                     "half": res["half"].isoformat() if res["half"] else None,
                     "full": res["full"].isoformat() if res["full"] else None}
        if not args.json:
            print(summarize("日本", asof, res))
            if args.verbose:
                print(detail(res))
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
