"""Convert a MEDS 0.3 dataset (e.g. from meds_etl_omop) into the MEDS 0.1.3 patient format femr 0.2.3 consumes.

Each output row is one patient (meds.patient_schema()): events grouped by time and sorted, measurements kept in their
original order within an event. Birth and death are renamed to meds 0.1.3's codes. Rows without a time go into the
birth event, because femr 0.2.3 can't handle untimed events and ignores static_measurements. Patients without exactly
one timed birth row are excluded, because femr.pat_utils.get_patient_birthdate prints the patient's events when it
fails.

Unless --no-training-transforms is given, this also applies the steps from femr's Stanford post-ETL pipeline
(femr/post_etl_pipelines/stanford.py) that don't depend on Stanford-specific metadata, so the input looks more like
clmbr-t-base's training data:
  - move_pre_birth: drop rows more than 30 days before birth, move the remaining pre-birth rows to birth
  - move_to_day_end: move rows at exactly midnight to 23:59 that day
  - switch_to_icd10cm: rename ICD10/ codes to ICD10CM/
  - remove_nones: drop a valueless row if the same code has a value on the same day
These steps from that pipeline are skipped:
  - delta_encode: the femr 0.2.3 processor already drops repeated tokens within a day
  - move_visit_start_to_first_event_start and move_billing_codes: they need Stanford visit and Clarity metadata

Only aggregate counts are printed, never rows, codes or IDs.

Usage: python tools/meds_to_femr023.py PATH_TO_MEDS_DATASET PATH_TO_OUTPUT [--no-training-transforms]

Load the result with datasets.Dataset.from_parquet(os.path.join(PATH_TO_OUTPUT, "data", "*.parquet")).
"""

import argparse
import collections
import glob
import os

import meds
import polars as pl
import pyarrow.parquet as pq

RENAMES = {"MEDS_BIRTH": meds.birth_code, "MEDS_DEATH": meds.death_code}


def convert(lf: pl.LazyFrame, training_transforms: bool) -> tuple[pl.LazyFrame, dict[str, pl.LazyFrame]]:
    """Returns (one row per patient, a LazyFrame of counts per name) for a MEDS 0.3 table of measurements."""
    if "text_value" not in lf.collect_schema().names():
        lf = lf.with_columns(pl.lit(None, dtype=pl.String).alias("text_value"))
    code = pl.col("code").cast(pl.String).replace(RENAMES)
    if training_transforms:
        code = code.str.replace(r"^ICD10/", "ICD10CM/")
    lf = lf.select(
        pl.col("subject_id").cast(pl.Int64).alias("patient_id"),
        pl.col("time").cast(pl.Datetime("us")),
        code,
        pl.col("numeric_value").cast(pl.Float32),
        pl.col("text_value").cast(pl.String),
    ).with_row_index("row")

    births = (
        lf.filter((pl.col("code") == meds.birth_code) & pl.col("time").is_not_null())
        .group_by("patient_id")
        .agg(pl.col("time").min().alias("birth"), pl.len().alias("births"))
    )
    patients = lf.select("patient_id").unique().join(births, on="patient_id", how="left")
    counts = {
        "patients in input": patients.select(pl.len()),
        "patients excluded, no timed birth": patients.filter(pl.col("births").is_null()).select(pl.len()),
        "patients excluded, more than one birth": patients.filter(pl.col("births") > 1).select(pl.len()),
        "measurements in input": lf.select(pl.len()),
    }

    lf = lf.join(births.filter(pl.col("births") == 1).select("patient_id", "birth"), on="patient_id")
    timeless = pl.col("time").is_null()
    counts["timeless measurements moved to birth"] = lf.filter(timeless).select(pl.len())
    lf = lf.with_columns(timeless.alias("timeless"), pl.col("time").fill_null(pl.col("birth")))

    if training_transforms:
        pre_birth = pl.col("time") < pl.col("birth")
        too_early = pl.col("time") < pl.col("birth") - pl.duration(days=30)
        counts["pre-birth measurements dropped"] = lf.filter(too_early).select(pl.len())
        counts["pre-birth measurements moved to birth"] = lf.filter(pre_birth & ~too_early).select(pl.len())
        lf = lf.filter(~too_early).with_columns(pl.max_horizontal("time", "birth").alias("time"))

        midnight = pl.col("time") == pl.col("time").dt.truncate("1d")
        counts["midnight measurements moved to 23:59"] = lf.filter(midnight).select(pl.len())
        lf = lf.with_columns(
            pl.when(midnight).then(pl.col("time") + pl.duration(days=1, minutes=-1)).otherwise("time").alias("time")
        )

        has_value = pl.col("numeric_value").is_not_null() | pl.col("text_value").is_not_null()
        redundant = ~has_value & has_value.any().over("patient_id", "code", pl.col("time").dt.date())
        counts["valueless measurements dropped"] = lf.filter(redundant).select(pl.len())
        lf = lf.filter(~redundant)

    counts["measurements written"] = lf.select(pl.len())
    measurement = pl.struct(
        "code",
        "text_value",
        "numeric_value",
        pl.lit(None, dtype=pl.Datetime("us")).alias("datetime_value"),
        pl.lit(None).alias("metadata"),
    )
    events = (
        lf.sort("patient_id", "time", "timeless", "row")
        .group_by("patient_id", "time", maintain_order=True)
        .agg(measurement.alias("measurements"))
    )
    counts["events written"] = events.select(pl.len())
    result = events.group_by("patient_id", maintain_order=True).agg(pl.struct("time", "measurements").alias("events"))
    counts["patients written"] = result.select(pl.len())
    return result, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meds_dataset", help="MEDS 0.3 dataset root, the directory containing data/")
    parser.add_argument("output", help="Directory to create; patients are written to output/data/")
    parser.add_argument("--no-training-transforms", action="store_true", help="Skip the Stanford post-ETL steps")
    parser.add_argument("--patients-per-row-group", type=int, default=1000)
    args = parser.parse_args()

    data_dir = os.path.join(args.meds_dataset, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    assert files, f"No parquet files under {data_dir}"
    os.makedirs(os.path.join(args.output, "data"))

    schema = meds.patient_schema()
    totals: collections.Counter = collections.Counter()
    seen_patients: set[int] = set()
    # MEDS shards by subject, so each file holds whole patients and can be converted on its own
    for i, fname in enumerate(files):
        patients, counts = convert(pl.scan_parquet(fname), not args.no_training_transforms)
        patients = patients.with_columns(pl.lit([], dtype=pl.List(pl.Null)).alias("static_measurements"))
        patients, *count_frames = pl.collect_all([patients.select(schema.names), *counts.values()])
        totals.update({name: frame.item() for name, frame in zip(counts, count_frames)})

        ids = set(patients["patient_id"].to_list())
        assert seen_patients.isdisjoint(ids), "Some patients span more than one input file"
        seen_patients |= ids

        # meds 0.1.3 uses 32-bit offsets, so cast in slices to keep e.g. note text under 2 GB per chunk
        table = patients.to_arrow()
        with pq.ParquetWriter(os.path.join(args.output, "data", f"{i:05d}.parquet"), schema) as writer:
            for start in range(0, table.num_rows, args.patients_per_row_group):
                writer.write_table(table.slice(start, args.patients_per_row_group).cast(schema))

    print(f"{len(files)} files converted into {os.path.join(args.output, 'data')}")
    width = max(map(len, totals))
    for name, count in totals.items():
        print(f"  {name:<{width}}  {count:,}")


if __name__ == "__main__":
    main()
