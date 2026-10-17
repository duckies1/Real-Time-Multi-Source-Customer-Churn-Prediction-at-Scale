"""Make a consistent, approximately 100 MiB user-hash sample of the source CSVs."""
import argparse
import csv
import hashlib
import os
import shutil

BASE = os.environ.get("DATA_DIR", "/dataset")
SOURCES = {
    "train": os.environ.get("TRAIN_FILE", f"{BASE}/train_v2.csv/data/churn_comp_refresh/train_v2.csv"),
    "transactions": os.environ.get("TRANSACTIONS_FILE", f"{BASE}/transactions_v2.csv/data/churn_comp_refresh/transactions_v2.csv"),
    "user_logs": os.environ.get("LOGS_FILE", f"{BASE}/user_logs_v2.csv/data/churn_comp_refresh/user_logs_v2.csv"),
    "members": os.environ.get("MEMBERS_FILE", f"{BASE}/members_v3.csv"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target-mib", type=float, default=100, help="target combined output size in MiB")
parser.add_argument("--output", default="/artifacts/kkbox-100mb", help="writable output directory")
args = parser.parse_args()

source_bytes = sum(os.path.getsize(path) for path in SOURCES.values())
fraction = min(args.target_mib * 1024 * 1024 / source_bytes, 1.0)
threshold = int(fraction * (1 << 64))
print(f"Source CSV bytes={source_bytes}; target_mib={args.target_mib}; deterministic_user_fraction={fraction:.6f}")

if os.path.exists(args.output):
    shutil.rmtree(args.output)
os.makedirs(args.output)

for name, source in SOURCES.items():
    destination = os.path.join(args.output, f"{name}.csv")
    kept = 0
    with open(source, "r", encoding="utf-8", newline="") as src, open(destination, "w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        try:
            id_col = header.index("msno")
        except ValueError as exc:
            raise RuntimeError(f"Expected an msno column in {source}") from exc
        writer.writerow(header)
        for row in reader:
            bucket = int.from_bytes(hashlib.md5(row[id_col].encode("utf-8")).digest()[:8], "big")
            if bucket < threshold:
                writer.writerow(row)
                kept += 1
    size = os.path.getsize(destination)
    print(f"{name}: rows={kept:,}; bytes={size:,}; path={destination}")

total = sum(os.path.getsize(os.path.join(args.output, f"{name}.csv")) for name in SOURCES)
print(f"SLICE_READY total_bytes={total:,} total_mib={total / (1024 * 1024):.2f} output={args.output}")
