#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rate_path_ois.py — OIS 版の利上げ・利下げ織り込み計算

rate_path.py と同じ階段フィット（solve_path）を、国債利回りの代わりに
OIS（翌日物金利スワップ）の固定金利へ適用する。

国債利回りには期間プレミアム・需給（担保としての品薄プレミアム等）が
混ざるが、OIS の固定金利は「その期間の翌日物金利の平均の見込み」とほぼ
一致するよう裁定が働くため、織り込みの読みは OIS のほうが素直になる。
代償として、OIS カーブの毎日の値は公式には公表されず、有料データが基本。
ここでは公開ページで日次値を出している BlueGamma（インターディーラー・
ブローカーの気配からカーブを構築する第三者）の表を使う。第三者集計値で
ある点には注意（README の「OIS 版」節）。

データ:
  米国 SOFR OIS : https://www.bluegamma.io/usd-swap-rates
  日本 TONA OIS : https://www.bluegamma.io/swap-rates/tona-swap-rates
  r0 は rate_path.py と同じ取得経路（米国 = FRED SOFR×365/360、
  日本 = 日銀 API の無担保コール翌日物。TONAR とほぼ同水準で推移する）

使い方:
  python3 rate_path_ois.py            # 日米とも計算
  python3 rate_path_ois.py --jp       # 日本のみ
  python3 rate_path_ois.py --compare  # 国債版（rate_path.py）も走らせて比較表示
  python3 rate_path_ois.py --verbose  # 会合ごとの内訳も表示

λ は既定で日米とも 6.25。OIS は年限の点が細かく、国債版の米国 λ=100 では
最初の会合の段差が平滑化で潰れてしまうため、小さめに置く。
"""

import argparse
import re
import sys
from datetime import date, datetime

import rate_path as rp

BG_US = "https://www.bluegamma.io/usd-swap-rates"
BG_JP = "https://www.bluegamma.io/swap-rates/tona-swap-rates"

TENOR_YEARS = {
    "1 Month": 1 / 12, "2 Month": 2 / 12, "3 Month": 3 / 12,
    "6 Month": 0.5, "9 Month": 0.75,
    "1 Year": 1.0, "2 Year": 2.0, "3 Year": 3.0,
}


def fetch_bluegamma_ois(url: str, max_years: float = 3.0):
    """BlueGamma のスワップレート表から (基準日, [(年限, レート%), ...]) を返す。

    表の "latest" 列が直近の引け値。ページ内の "Last update:" の日付を基準日にする。
    """
    html = rp._get(url).decode("utf-8", errors="ignore")
    m = re.search(r"Last update:</strong>\s*([A-Za-z]+ \d{1,2}, \d{4})", html)
    asof = datetime.strptime(m.group(1), "%B %d, %Y").date() if m else date.today()
    obs = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        th = re.search(r"<th[^>]*>(.*?)</th>", row, re.S)
        td = re.search(r'<td class="gr-c-latest"[^>]*>\s*([\d.]+)%', row)
        if not th or not td:
            continue
        label = re.sub(r"<[^>]+>", "", th.group(1)).strip()
        if label in TENOR_YEARS and TENOR_YEARS[label] <= max_years:
            obs.append((TENOR_YEARS[label], float(td.group(1))))
    if not obs:
        raise RuntimeError(f"BlueGamma の表を解析できませんでした: {url}")
    return asof, sorted(obs)


def run_us_ois(lam: float = 6.25, verbose: bool = False) -> dict:
    asof, obs = fetch_bluegamma_ois(BG_US)
    obs = [(T, y * 365 / 360) for T, y in obs]  # USD OIS は ACT/360 建て
    _, r0, _ = rp.fetch_us()                    # SOFR×365/360（rate_path と同じ基準）
    res = rp.solve_path(r0, obs, rp.meeting_dates(rp.FOMC_DATES, asof), asof, lam)
    print(rp.summarize("米国 OIS", asof, res))
    if verbose:
        print(rp.detail(res))
    return res


def run_jp_ois(lam: float = 6.25, verbose: bool = False) -> dict:
    asof, obs = fetch_bluegamma_ois(BG_JP)
    _, r0 = rp.fetch_jp_r0()
    res = rp.solve_path(r0, obs, rp.meeting_dates(rp.BOJ_DATES, asof), asof, lam)
    print(rp.summarize("日本 OIS", asof, res))
    if verbose:
        print(rp.detail(res))
    return res


def run_us_bond() -> dict:
    asof, r0, obs = rp.fetch_us()
    return rp.solve_path(r0, obs, rp.meeting_dates(rp.FOMC_DATES, asof), asof, 100.0)


def run_jp_bond() -> dict:
    asof, r0, obs = rp.fetch_jp()
    return rp.solve_path(r0, obs, rp.meeting_dates(rp.BOJ_DATES, asof), asof, 6.25)


def _row(label: str, res: dict) -> str:
    half = res["half"].isoformat() if res["half"] else "-"
    full = res["full"].isoformat() if res["full"] else "-"
    nxt = res["meetings"][0].isoformat() if res["meetings"] else "-"
    return (f"  {label:<8} 次回 {nxt} の p = {res['p'][0]*100:>5.0f}%   "
            f"半分 {half}   見込みきる {full}   1年 {res['n_1y']:+.1f} 回")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OIS 版の政策金利織り込み計算")
    ap.add_argument("--us", action="store_true", help="米国のみ")
    ap.add_argument("--jp", action="store_true", help="日本のみ")
    ap.add_argument("--compare", action="store_true",
                    help="国債版（rate_path.py）も実行して比較表示")
    ap.add_argument("--verbose", action="store_true", help="会合ごとの内訳も表示")
    ap.add_argument("--lam", type=float, default=6.25, help="平滑化 λ（既定 6.25）")
    args = ap.parse_args(argv)

    both = not (args.us or args.jp)
    if both or args.us:
        res_us = run_us_ois(args.lam, args.verbose)
        if args.compare:
            print("  [比較] 米国")
            print(_row("国債版", run_us_bond()))
            print(_row("OIS 版", res_us))
    if both or args.jp:
        res_jp = run_jp_ois(args.lam, args.verbose)
        if args.compare:
            print("  [比較] 日本")
            print(_row("国債版", run_jp_bond()))
            print(_row("OIS 版", res_jp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
