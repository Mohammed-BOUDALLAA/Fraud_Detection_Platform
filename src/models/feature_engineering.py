import gc
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import RAW_DIR, PROC_DIR, MODEL_DIR


class FeatureBuilder:
    def fit(self, frame):
        self.card_means = (
            frame.groupby("card1").TransactionAmt.mean().to_dict()
        )
        self.card_counts = frame.card1.value_counts().to_dict()
        self.global_mean = float(frame.TransactionAmt.mean())

        self.categories = {
            c: {
                value: i
                for i, value in enumerate(
                    sorted(frame[c].dropna().astype(str).unique())
                )
            }
            for c in frame.select_dtypes(
                include=["object", "string", "category"]
            ).columns
            if c not in ("TransactionID", "isFraud")
        }

        # Only the schema is needed here, not a transformed training set.
        self.feature_cols = self._engineer(frame.iloc[:1]).columns.tolist()
        return self

    def _engineer(self, frame):
        # Construct output column by column; never copy the entire input.
        out = {}
        excluded = {"TransactionID", "isFraud", "TransactionDT"}

        def clean(values):
            values = pd.to_numeric(values, errors="raise")
            return (
                values.astype("float32")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(-999)
            )

        for c in frame.columns:
            if c in excluded:
                continue

            if c in self.categories:
                values = (
                    frame[c]
                    .astype("string")
                    .map(self.categories[c])
                    .fillna(-1)
                )
                out[c] = values.astype("float32")
            elif pd.api.types.is_numeric_dtype(frame[c]):
                out[c] = clean(frame[c])

        # Preserve support for missing categorical inference columns.
        for c in self.categories:
            if c not in out:
                out[c] = np.float32(-1)

        amount = pd.to_numeric(frame["TransactionAmt"], errors="raise")
        if (amount < 0).any():
            raise ValueError("TransactionAmt must be non-negative")

        card_mean = (
            frame["card1"].map(self.card_means).fillna(self.global_mean)
        )
        hour = (frame["TransactionDT"] // 3600) % 24

        out["TransactionAmt_log"] = clean(np.log1p(amount))
        out["amt_vs_card_mean"] = clean(amount / (card_mean + 1))
        out["card1_count"] = clean(
            frame["card1"].map(self.card_counts).fillna(0)
        )
        out["hour_sin"] = clean(np.sin(2 * np.pi * hour / 24))
        out["hour_cos"] = clean(np.cos(2 * np.pi * hour / 24))
        out["day_of_week"] = clean(
            (frame["TransactionDT"] // 86400) % 7
        )
        return pd.DataFrame(out, index=frame.index)

    def transform(self, frame):
        return self._engineer(frame).reindex(
            columns=self.feature_cols, fill_value=np.float32(-999)
        )


def read_csv_compact(path, chunk_size=25000):
    """Read in chunks, reducing floating-point storage before concatenation."""
    chunks = []

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        for c in chunk.select_dtypes(include=["float64"]).columns:
            # Keep identifiers and timestamps at their original precision.
            if c not in ("TransactionID", "TransactionDT"):
                chunk[c] = chunk[c].astype("float32")
        chunks.append(chunk)

    if not chunks:
        raise ValueError(f"Empty CSV: {path}")

    return pd.concat(chunks, ignore_index=True)


def load_raw_data(raw_dir=RAW_DIR):
    tx_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    if not tx_path.exists() or not id_path.exists():
        raise FileNotFoundError(
            f"Place BOTH IEEE-CIS training CSVs in {raw_dir}"
        )

    print("Loading transactions...", flush=True)
    tx = read_csv_compact(tx_path)

    print("Loading identity data...", flush=True)
    identity = read_csv_compact(id_path)

    required = {
        "TransactionID", "TransactionDT",
        "TransactionAmt", "card1", "isFraud",
    }
    missing = required - set(tx.columns)
    if missing:
        raise ValueError(f"Missing transaction columns: {missing}")
    if "TransactionID" not in identity:
        raise ValueError("Identity CSV is missing TransactionID")

    if (
        tx.TransactionID.isna().any()
        or identity.TransactionID.isna().any()
    ):
        raise ValueError("Missing TransactionID")

    if (
        tx.TransactionID.duplicated().any()
        or identity.TransactionID.duplicated().any()
    ):
        raise ValueError("Duplicate TransactionID in input")

    if not tx.isFraud.isin([0, 1]).all():
        raise ValueError("isFraud must be binary with no missing values")
    if tx.TransactionDT.isna().any():
        raise ValueError("Missing TransactionDT")

    print("Merging data...", flush=True)
    frame = tx.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )
    del tx, identity
    gc.collect()

    frame = frame.sort_values("TransactionDT", kind="stable")
    frame.reset_index(drop=True, inplace=True)
    return frame


def build(
    raw_dir=RAW_DIR,
    provenance="IEEE-CIS user-provided training CSVs",
    batch_size=10000,
):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_raw_data(raw_dir)
    n = len(frame)
    if n < 3:
        raise ValueError("Not enough transactions")

    times = frame.TransactionDT.to_numpy()
    a = int(np.searchsorted(times, times[int(n * .7)], side="left"))
    b = int(np.searchsorted(times, times[int(n * .85)], side="left"))

    if not 0 < a < b < n:
        raise ValueError(
            "Need enough distinct timestamps for a 70/15/15 temporal split"
        )

    print(f"Fitting preprocessing on {a:,} training rows...", flush=True)
    builder = FeatureBuilder().fit(frame.iloc[:a])

    output = PROC_DIR / "features.parquet"
    temporary = PROC_DIR / "features.building.parquet"
    writer = None

    try:
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = frame.iloc[start:stop]

            features = builder.transform(batch)
            features["isFraud"] = batch.isFraud.astype("int8")

            table = pa.Table.from_pandas(
                features, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="snappy"
                )

            writer.write_table(table)
            del batch, features, table
            print(f"Processed {stop:,}/{n:,} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()

    temporary.replace(output)

    frame[["TransactionID", "TransactionDT"]].to_parquet(
        PROC_DIR / "row_metadata.parquet", index=False
    )
    joblib.dump(builder, MODEL_DIR / "preprocessor.pkl")
    joblib.dump(builder.feature_cols, MODEL_DIR / "feature_cols.pkl")

    manifest = {
        "provenance": provenance,
        "synthetic": provenance.startswith("SYNTHETIC"),
        "rows": n,
        "features": len(builder.feature_cols),
        "train_end": a,
        "validation_end": b,
        "split": (
            "chronological 70/15/15, timestamp boundaries; "
            "preprocessing fit on train only"
        ),
    }
    (PROC_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    # Keep serialized class imports valid in training and the API.
    from src.models.feature_engineering import build as run_build

    run_build()