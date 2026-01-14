#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

PATTERNS = ["summary_D4*.csv"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_D4_csv.py <dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.exists():
        raise FileNotFoundError(root)

    csv_paths = []
    for pat in PATTERNS:
        csv_paths.extend(root.rglob(pat))

    # dedupe + sort
    csv_paths = sorted(set(csv_paths))

    if not csv_paths:
        print(f"No matching CSV found under: {root}")
        print("Tried patterns:", ", ".join(PATTERNS))
        sys.exit(0)

    rows = []
    fieldnames = None

    for csv_path in csv_paths:
        tag = csv_path.parent.name
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = ["tag", "source_path"] + reader.fieldnames

            for row in reader:
                out = {"tag": tag, "source_path": str(csv_path)}
                out.update(row)
                rows.append(out)

    if not rows:
        print("CSV files found, but no rows could be read.")
        sys.exit(0)

    out_path = root.parent / "summary_D4_all.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Collected {len(rows)} rows from {len(csv_paths)} files")
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
