"""Download the open DE-LU day-ahead price series used in this repo.

Source: energy-charts.info API, data from Bundesnetzagentur | SMARD.de, CC BY 4.0.
No API key required.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://api.energy-charts.info/price?bzn={bzn}&start={start}&end={end}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bzn", default="DE-LU")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-10-18")
    ap.add_argument("--out", default="data/raw/de_lu_price_raw.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(bzn=args.bzn, start=args.start, end=args.end)
    print("GET", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    out.write_bytes(payload)
    print(f"saved {len(payload) / 1024:.0f} KB -> {out}")


if __name__ == "__main__":
    main()
