"""Break down why clmbr-t-base can't map parts of a MEDS dataset, with a suggested fix for each reason.

tools/clmbr_vocab_coverage.py measures how much of a dataset maps to a model token. This script puts each unmapped
measurement into one reason (see REASONS), following the branches of the flat FEMRTokenizer.get_feature_codes, and
each reason has its own fix. It also flags mapped numeric codes whose values pile into the model's lowest or highest
bin, which suggests different units than the model's training data.

Works on MEDS 0.3 datasets and on the MEDS 0.1.3 output of tools/meds_to_femr023.py. The converter's output is closer
to what the model sees. Codes are renamed the way the converter does by default (birth, death, ICD10/ to ICD10CM/), so
both give the same codes.

Shares are reported three ways:
  - measurements: rows in the data
  - day units: unique (patient, day, token) for mapped measurements and (patient, day, code) for unmapped ones. The
    femr 0.2.3 processor drops repeated tokens within a day, so this is closer to what the model would see.
  - recent window: day units on the days holding each patient's last --window-tokens mapped tokens. compute_features
    only uses the last tokens_per_batch (default 1024) tokens before a label, so this approximates a label at the end
    of the record.

The Maps-to check (A3) needs metadata/codes.parquet with parent_codes, which meds_etl_omop writes. The converter doesn't
copy metadata, so pass --codes-metadata RAW_DATASET/metadata/codes.parquet when running on its output.

Printed output is shares only, plus counts about the model's public vocabulary. The patients-affected column is the
most sensitive on small cohorts. With --details DIR, per-code tables are written to DIR and not printed:
  - unmapped_codes.csv: each unmapped (code, kind) with its reason, counts and hints (the matching model code, the
    Maps-to target, the model's text values or bins for that code)
  - numeric_bin_occupancy.csv: for each mapped numeric code, how its values spread across the model's bins
  - vocabularies.csv: coverage per vocabulary (code prefix) and whether the model has that vocabulary
These name your codes and vocabularies, which can identify the dataset, so handle them like the data itself.

Usage: python tools/clmbr_unmapped_diagnosis.py PATH_TO_MEDS_DATASET [--codes-metadata PATH] [--details DIR]
"""

import argparse
import glob
import os

import polars as pl

# Short id: (what went wrong, suggested fix)
REASONS = {
    "A1": ("Unknown code, and the model has no codes from its vocabulary", "Remap to codes the model knows, or accept"),
    "A2": (
        "Unknown code, but a model code matches after normalizing case, whitespace, dots or leading zeros",
        "Normalize the code format in the converter",
    ),
    "A3": (
        "Unknown code, but its Maps-to target is a model code of the same kind",
        "Swap in the Maps-to target in the converter",
    ),
    "A4": (
        "Unknown code from a vocabulary the model has",
        "Usually rare or newer codes: accept, or remap via an ancestor",
    ),
    "B1": ("Numeric value, but the model has no value bins for the code", "Emit as the bare code"),
    "B2": ("Numeric value outside all of the code's bins", "Check units and the model's bins for the code"),
    "B3": ("Numeric value is NaN", "Drop the value, or emit as the bare code"),
    "B4a": (
        "Text value matches one of the code's model values only after normalizing case and whitespace",
        "Normalize the text in the converter",
    ),
    "B4b": ("Text value isn't among the code's model values", "Accept, or map to a value the model knows"),
    "B4c": (
        "Text value, but the model has no text values for the code",
        "Emit as the bare code, or parse numbers out of the text",
    ),
    "B5": ("No value, but the model knows the code only with values", "Check whether the ETL drops values, or accept"),
}

# The renames tools/meds_to_femr023.py does by default
RENAMES = {"MEDS_BIRTH": "SNOMED/184099003", "MEDS_DEATH": "SNOMED/419620001"}


def rename(code: pl.Expr) -> pl.Expr:
    return code.cast(pl.String).replace(RENAMES).str.replace(r"^ICD10/", "ICD10CM/")


def vocabulary(code: pl.Expr) -> pl.Expr:
    return code.str.split("/").list.first()


def code_key(code: pl.Expr) -> pl.Expr:
    """Normalizes case, whitespace, dots and leading zeros, to find codes that differ from the model's only in format"""
    prefix = code.str.extract(r"^([^/]*)/", 1).str.strip_chars().str.to_uppercase()
    suffix = code.str.replace(r"^[^/]*/", "").str.to_uppercase().str.replace_all(r"[.\s]", "")
    suffix = suffix.str.replace(r"^0+(.)", "${1}")
    return pl.when(prefix.is_null()).then(suffix).otherwise(prefix + "/" + suffix)


def text_key(text: pl.Expr) -> pl.Expr:
    return text.str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ")


def model_tables(code_lookup, numeric_lookup, string_lookup) -> dict[str, pl.DataFrame]:
    """Returns the model's vocabulary as tables: plain codes, numeric bins, text values, and what's known per code."""
    plain = pl.DataFrame(
        {"code": list(code_lookup), "token": list(code_lookup.values())}, schema={"code": pl.String, "token": pl.Int64}
    )
    bins = pl.DataFrame(
        [(code, start, end, i) for code, entries in numeric_lookup.items() for start, end, i in entries],
        schema={"code": pl.String, "start": pl.Float64, "end": pl.Float64, "token": pl.Int64},
        orient="row",
    )
    text = pl.DataFrame(
        [(code, text, i) for (code, text), i in string_lookup.items()],
        schema={"code": pl.String, "text_value": pl.String, "token": pl.Int64},
        orient="row",
    )
    kinds = (
        pl.concat(
            [
                plain.select("code", has_plain=pl.lit(True)),
                bins.select("code", has_bins=pl.lit(True)),
                text.select("code", has_text=pl.lit(True)),
            ],
            how="diagonal",
        )
        .group_by("code")
        .agg(pl.col("has_plain", "has_bins", "has_text").any())
    )
    return {
        "plain": plain,
        "bins": bins,
        "text": text,
        "text_keys": text.select("code", text_key(pl.col("text_value")).alias("text_key")).unique(),
        "kinds": kinds,
        "code_keys": kinds.group_by(code_key(pl.col("code")).alias("key")).agg(pl.col("code").min().alias("match")),
        "vocabularies": kinds.select(vocabulary(pl.col("code")).alias("vocabulary")).unique(),
    }


def maps_to_targets(path: str, model: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Returns, per code, a Maps-to target the model knows as a plain code, with bins, and with text values."""
    return (
        pl.scan_parquet(path)
        .select(rename(pl.col("code")), pl.col("parent_codes").alias("target"))
        .explode("target")
        .with_columns(rename(pl.col("target")))
        .join(model["kinds"].lazy(), left_on="target", right_on="code")
        .group_by("code")
        .agg(
            *[pl.col("target").filter(pl.col(f"has_{k}")).min().alias(f"target_{k}") for k in ("plain", "bins", "text")]
        )
        .collect()
    )


def measurements(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Returns (subject_id, date, code, numeric_value, text_value) per measurement, from either MEDS version."""
    if "events" in lf.collect_schema().names():
        # One row per patient (MEDS 0.1.3, e.g. from tools/meds_to_femr023.py), so flatten to one row per measurement
        lf = (
            lf.select(pl.col("patient_id").alias("subject_id"), "events")
            .explode("events")
            .select("subject_id", pl.col("events").struct.field("time", "measurements"))
            .explode("measurements")
            .select("subject_id", "time", pl.col("measurements").struct.field("code", "numeric_value", "text_value"))
        )
    if "text_value" not in lf.collect_schema().names():
        lf = lf.with_columns(pl.lit(None, dtype=pl.String).alias("text_value"))
    # Untimed rows go to the oldest day, as the converter moves them to birth
    return lf.select(
        "subject_id",
        pl.col("time").dt.date().fill_null(pl.date(1, 1, 1)).alias("date"),
        rename(pl.col("code")),
        pl.col("numeric_value").cast(pl.Float64),
        pl.col("text_value").cast(pl.String),
    ).with_row_index("row")


def classify(lf: pl.LazyFrame, model: dict[str, pl.DataFrame], targets: pl.DataFrame | None) -> pl.LazyFrame:
    """Adds kind, token (null if unmapped), reason ("mapped" or a REASONS id) and hint to each measurement."""
    kind = (
        pl.when(pl.col("numeric_value").is_not_null())
        .then(pl.lit("numeric"))
        .when(pl.col("text_value").is_not_null())
        .then(pl.lit("text"))
        .otherwise(pl.lit("code"))
    )
    lf = lf.with_columns(kind.alias("kind"))

    # The same lookups as the flat FEMRTokenizer.get_feature_codes
    in_bin = pl.col("numeric_value").is_between("start", "end", closed="left") & ~pl.col("numeric_value").is_nan()
    tokens = pl.concat(
        [
            lf.filter(pl.col("kind") == "numeric").join(model["bins"].lazy(), on="code").filter(in_bin),
            lf.filter(pl.col("kind") == "text").join(model["text"].lazy(), on=["code", "text_value"]),
            lf.filter(pl.col("kind") == "code").join(model["plain"].lazy(), on="code"),
        ],
        how="diagonal",
    ).select("row", "token")

    info = (
        lf.select("code")
        .unique()
        .join(model["kinds"].lazy(), on="code", how="left")
        .with_columns(pl.col("has_plain", "has_bins", "has_text").fill_null(False))
        .join(model["code_keys"].lazy(), left_on=code_key(pl.col("code")), right_on="key", how="left")
        .with_columns(vocabulary(pl.col("code")).alias("vocabulary"))
        .join(model["vocabularies"].lazy().with_columns(known_vocabulary=pl.lit(True)), on="vocabulary", how="left")
        .with_columns(pl.col("known_vocabulary").fill_null(False))
    )
    if targets is not None:
        info = info.join(targets.lazy(), on="code", how="left")
    else:
        info = info.with_columns(pl.lit(None, dtype=pl.String).alias(f"target_{k}") for k in ("plain", "bins", "text"))

    lf = (
        lf.join(tokens, on="row", how="left")
        .join(info, on="code", how="left")
        .with_columns(text_key(pl.col("text_value")).alias("text_key"))
        .join(
            model["text_keys"].lazy().with_columns(text_format=pl.lit(True)),
            on=["code", "text_key"],
            how="left",
        )
    )

    kind = pl.col("kind")
    known = pl.col("has_plain") | pl.col("has_bins") | pl.col("has_text")
    target = (
        pl.when(kind == "numeric")
        .then(pl.col("target_bins"))
        .when(kind == "text")
        .then(pl.col("target_text"))
        .otherwise(pl.col("target_plain"))
    )
    reason = (
        pl.when(pl.col("token").is_not_null())
        .then(pl.lit("mapped"))
        .when((kind == "numeric") & pl.col("numeric_value").is_nan())
        .then(pl.lit("B3"))
        .when((kind == "numeric") & pl.col("has_bins"))
        .then(pl.lit("B2"))
        .when((kind == "numeric") & known)
        .then(pl.lit("B1"))
        .when((kind == "text") & pl.col("text_format").fill_null(False))
        .then(pl.lit("B4a"))
        .when((kind == "text") & pl.col("has_text"))
        .then(pl.lit("B4b"))
        .when((kind == "text") & known)
        .then(pl.lit("B4c"))
        .when((kind == "code") & known)
        .then(pl.lit("B5"))
        .when(pl.col("match").is_not_null())
        .then(pl.lit("A2"))
        .when(target.is_not_null())
        .then(pl.lit("A3"))
        .when(~pl.col("known_vocabulary"))
        .then(pl.lit("A1"))
        .otherwise(pl.lit("A4"))
    )
    hint = pl.when(reason == "A2").then(pl.col("match")).when(reason == "A3").then(target)
    return lf.select("row", "subject_id", "date", "code", "kind", "token", reason.alias("reason"), hint.alias("hint"))


def summarize(m: pl.LazyFrame, window_tokens: int) -> dict[str, pl.LazyFrame]:
    """Returns per-reason, per-code, per-bin and per-vocabulary counts for one file of classified measurements."""
    unmapped = pl.col("token").is_null()
    units = m.select("subject_id", "date", "reason", "kind", "code", "token").unique()

    # The days holding each patient's last window_tokens mapped tokens
    window = (
        units.group_by("subject_id", "date")
        .agg((~unmapped).sum().alias("tokens"))
        .sort("subject_id", "date", descending=[False, True])
        .with_columns((pl.col("tokens").cum_sum().over("subject_id") - pl.col("tokens")).alias("later_tokens"))
        .filter(pl.col("later_tokens") < window_tokens)
        .select("subject_id", "date", in_window=pl.lit(True))
    )
    units = units.join(window, on=["subject_id", "date"], how="left").with_columns(pl.col("in_window").fill_null(False))
    unit_counts = [pl.len().alias("units"), pl.col("in_window").sum().alias("window_units")]

    return {
        "patients": m.select(pl.col("subject_id").n_unique().alias("patients")),
        "reasons": m.group_by("reason")
        .agg(pl.len().alias("rows"), pl.col("subject_id").n_unique().alias("patients"))
        .join(units.group_by("reason").agg(unit_counts), on="reason"),
        "codes": m.filter(unmapped)
        .group_by("code", "kind", "reason", "hint")
        .agg(pl.len().alias("rows"), pl.col("subject_id").n_unique().alias("patients"))
        .join(
            units.filter(unmapped).group_by("code", "kind", "reason").agg(unit_counts), on=["code", "kind", "reason"]
        ),
        "bins": m.filter((pl.col("kind") == "numeric") & ~unmapped)
        .group_by("code", "token")
        .agg(pl.len().alias("rows")),
        "vocabularies": m.group_by(vocabulary(pl.col("code")).alias("vocabulary")).agg(
            pl.len().alias("rows"), (~unmapped).sum().alias("mapped_rows")
        ),
    }


def bin_occupancy(counts: pl.DataFrame, bins: pl.DataFrame, min_rows: int, edge_share: float) -> pl.DataFrame:
    """Returns, per mapped numeric code, its values' shares in the lowest bin, the highest bin and the fullest bin."""
    edges = bins.with_columns(
        (pl.col("start") == pl.col("start").min().over("code")).alias("lowest"),
        (pl.col("start") == pl.col("start").max().over("code")).alias("highest"),
    )
    share = pl.col("rows") / pl.col("rows").sum()
    return (
        edges.join(counts, on=["code", "token"], how="left")
        .with_columns(pl.col("rows").fill_null(0))
        .group_by("code")
        .agg(
            pl.len().alias("bins"),
            pl.col("rows").sum(),
            share.filter(pl.col("lowest")).sum().alias("share_lowest_bin"),
            share.filter(pl.col("highest")).sum().alias("share_highest_bin"),
            share.max().alias("share_fullest_bin"),
        )
        .filter(pl.col("rows") > 0)
        .with_columns(
            (
                (pl.col("bins") >= 2)
                & (pl.col("rows") >= min_rows)
                & (pl.max_horizontal("share_lowest_bin", "share_highest_bin") > edge_share)
            ).alias("flagged")
        )
        .sort("rows", descending=True)
    )


def percent(expr: pl.Expr) -> pl.Expr:
    return pl.format("{}%", (expr * 100).round(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meds_dataset", help="MEDS dataset root, the directory containing data/")
    parser.add_argument("--model", default="StanfordShahLab/clmbr-t-base")
    parser.add_argument("--codes-metadata", help="codes.parquet with parent_codes (default: metadata/ in the dataset)")
    parser.add_argument("--details", help="Directory to write per-code tables to (they name your codes)")
    parser.add_argument("--window-tokens", type=int, default=1024, help="Tokens in the recent window")
    parser.add_argument("--min-rows", type=int, default=100, help="Mapped values a code needs for the bin check")
    parser.add_argument("--edge-share", type=float, default=0.5, help="Edge-bin share above which a code is flagged")
    parser.add_argument("--max-files", type=int, help="Only read the first N data files")
    args = parser.parse_args()

    import femr.models.tokenizer

    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(args.model)
    assert not tokenizer.is_hierarchical, "This script mirrors the flat tokenizer only"
    model = model_tables(tokenizer.code_lookup, tokenizer.numeric_lookup, tokenizer.string_lookup)

    files = sorted(glob.glob(os.path.join(args.meds_dataset, "data", "**", "*.parquet"), recursive=True))
    files = files[: args.max_files]
    assert files, f"No parquet files under {os.path.join(args.meds_dataset, 'data')}"

    metadata = args.codes_metadata or os.path.join(args.meds_dataset, "metadata", "codes.parquet")
    targets = None
    if not os.path.exists(metadata):
        note = f"{metadata} not found, so the Maps-to check (A3) was skipped"
    elif "parent_codes" not in pl.scan_parquet(metadata).collect_schema().names():
        note = f"{metadata} has no parent_codes, so the Maps-to check (A3) was skipped"
    else:
        targets = maps_to_targets(metadata, model)
        note = f"Maps-to targets from {metadata}: {len(targets):,} codes have one the model knows"

    # MEDS shards by subject, so each file holds whole patients and counts can be summed across files
    parts: dict[str, list[pl.DataFrame]] = {}
    for fname in files:
        summary = summarize(classify(measurements(pl.scan_parquet(fname)), model, targets), args.window_tokens)
        for name, frame in zip(summary, pl.collect_all(list(summary.values()))):
            parts.setdefault(name, []).append(frame)

    def total(name: str, *keys: str) -> pl.DataFrame:
        return pl.concat(parts[name]).group_by(keys).agg(pl.exclude(keys).sum())

    patients = pl.concat(parts["patients"])["patients"].sum()
    reasons = total("reasons", "reason")
    codes = total("codes", "code", "kind", "reason", "hint")
    occupancy = bin_occupancy(total("bins", "code", "token"), model["bins"], args.min_rows, args.edge_share)
    vocabularies = total("vocabularies", "vocabulary")

    # Model vocabulary, which is public
    bins_per_code = model["bins"].group_by("code").len()["len"]
    print(
        f"Model vocabulary: {len(model['plain']):,} plain codes, {len(bins_per_code):,} numeric codes with"
        f" {len(model['bins']):,} bins, {len(model['text']):,} (code, text) values"
    )
    if len(bins_per_code):
        one_bin = (bins_per_code == 1).mean()
        print(
            f"  Bins per numeric code: median {bins_per_code.median():g}, {one_bin:.0%} of numeric codes have one bin"
        )
        if one_bin > 0.5:
            print(
                "  Most numeric codes have a single bin, so values outside it can't map (B2) and the bin check below"
                " skips them"
            )
    print(note)

    mapped = reasons.filter(pl.col("reason") == "mapped")
    columns = ("rows", "units", "window_units")
    totals = reasons.select(pl.col(columns).sum()).row(0, named=True)
    unmapped = reasons.filter(pl.col("reason") != "mapped").select(pl.col(columns).sum()).row(0, named=True)
    print(f"\n{len(files)} files. Share that maps to a model token:")
    for column, label in zip(
        columns, ("measurements", "day units", f"day units in the last {args.window_tokens} tokens")
    ):
        print(f"  {label}: {mapped[column].sum() / max(1, totals[column]):.1%}")

    table = (
        pl.DataFrame({"reason": list(REASONS)})
        .join(reasons, on="reason", how="left")
        .fill_null(0)
        .select(
            "reason",
            *[percent(pl.col(c) / max(1, unmapped[c])).alias(c) for c in columns],
            percent(pl.col("patients") / max(1, patients)).alias("patients"),
        )
        .rename({"rows": "measurements", "units": "day units", "window_units": "window units"})
    )
    with pl.Config(tbl_rows=len(REASONS), tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(
            "\nWhy measurements don't map (share of unmapped measurements, day units and window units; share of all"
            " patients with any):",
            table,
            sep="\n",
        )
    for reason, (description, fix) in REASONS.items():
        print(f"  {reason:<4} {description}. Fix: {fix}")

    checked = occupancy.filter((pl.col("bins") >= 2) & (pl.col("rows") >= args.min_rows))
    flagged = checked.filter("flagged")
    print(
        f"\nBin check: of the numeric codes with at least {args.min_rows} mapped values and two or more bins,"
        f" {len(flagged) / max(1, len(checked)):.0%} have over {args.edge_share:.0%} of their values in the lowest or"
        f" highest bin. They hold {flagged['rows'].sum() / max(1, occupancy['rows'].sum()):.1%} of mapped numeric"
        " measurements. Bins are deciles of the training data, so matching units put about 1/bins of a code's values"
        " in each bin. Population differences shift this somewhat; a unit mismatch puts nearly all values in one edge"
        " bin."
    )

    if args.details:
        os.makedirs(args.details, exist_ok=True)
        text_values = (
            model["text"]
            .group_by("code")
            .agg(
                pl.len().alias("model_text_values"),
                pl.col("text_value").sort().head(3).str.join(" | ").alias("model_text_examples"),
            )
        )
        bin_range = (
            model["bins"]
            .group_by("code")
            .agg(
                pl.format("{} bins, [{}, {})", pl.len(), pl.col("start").min(), pl.col("end").max()).alias("model_bins")
            )
        )
        codes = (
            codes.join(text_values, on="code", how="left")
            .join(bin_range, on="code", how="left")
            .sort("units", descending=True)
        )
        vocabularies = (
            vocabularies.with_columns((pl.col("mapped_rows") / pl.col("rows")).alias("coverage"))
            .join(model["vocabularies"].with_columns(in_model=pl.lit(True)), on="vocabulary", how="left")
            .with_columns(pl.col("in_model").fill_null(False))
            .sort("rows", descending=True)
        )
        for name, frame in [
            ("unmapped_codes", codes),
            ("numeric_bin_occupancy", occupancy),
            ("vocabularies", vocabularies),
        ]:
            frame.write_csv(os.path.join(args.details, f"{name}.csv"))
        print(f"\nPer-code tables written to {args.details}")


if __name__ == "__main__":
    main()
