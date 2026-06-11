import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SHARDS = REPO / "results" / "baselines"
OUT = SHARDS / "sweep.parquet"


def main():
    parts = []
    for p in sorted(SHARDS.glob("*/sweep.parquet")):
        df = pd.read_parquet(p)
        parts.append(df)
        print(f"loaded {p}: {df.shape}")
    if not parts:
        print("no shards found")
        return 1
    merged = pd.concat(parts, ignore_index=True)
    merged.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {merged.shape}, methods={sorted(merged['method'].unique())}, scenarios={sorted(merged['scenario'].unique())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
