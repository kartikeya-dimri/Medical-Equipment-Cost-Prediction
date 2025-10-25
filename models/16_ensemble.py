"""models/16_ensemble.py

Create an ensemble CSV by averaging Transport_Cost from two model outputs.

Usage (defaults assume your files live in the `output/` folder):
	python models/16_ensemble.py
	python models/16_ensemble.py --a output/gradient_boosting_tuned_3.csv \
		--b output/random_forest_tuned_3.csv --out output/ensemble.csv

The script merges on `Hospital_Id` (outer join), averages values when both
are present, and writes `Hospital_Id,Transport_Cost` to the output path.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
	import pandas as pd
except Exception:
	print("This script requires pandas. Install with: python -m pip install pandas", file=sys.stderr)
	raise


def load_df(path: str, col_name: str) -> pd.DataFrame:
	df = pd.read_csv(path, dtype={"Hospital_Id": str})
	if "Transport_Cost" not in df.columns:
		# try to detect a cost-like column
		candidates = [c for c in df.columns if c.lower().replace(" ", "_") in ("transport_cost", "cost", "transportcost")]
		if candidates:
			df = df.rename(columns={candidates[0]: "Transport_Cost"})
		else:
			raise KeyError(f"'Transport_Cost' column not found in {path}. Columns: {list(df.columns)}")
	df = df[["Hospital_Id", "Transport_Cost"]].copy()
	df["Transport_Cost"] = pd.to_numeric(df["Transport_Cost"], errors="coerce")
	return df.rename(columns={"Transport_Cost": col_name})


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description="Average Transport_Cost between two model CSV outputs.")
	p.add_argument("--a", help="first CSV file", default=os.path.join("output", "gradient_boosting_tuned_3.csv"))
	p.add_argument("--b", help="second CSV file", default=os.path.join("output", "random_forest_tuned_3.csv"))
	p.add_argument("--out", help="output CSV path", default=os.path.join("output", "ensemble.csv"))
	args = p.parse_args(argv)

	df_a = load_df(args.a, "cost_a")
	df_b = load_df(args.b, "cost_b")

	merged = pd.merge(df_a, df_b, on="Hospital_Id", how="outer")

	# average available values; if only one exists, mean(...) returns that value
	merged["Transport_Cost"] = merged[["cost_a", "cost_b"]].mean(axis=1, skipna=True)

	result = merged[["Hospital_Id", "Transport_Cost"]].copy()

	out_dir = os.path.dirname(args.out)
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)

	result.to_csv(args.out, index=False)
	print(f"Wrote ensemble CSV to: {args.out} (rows: {len(result)})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

