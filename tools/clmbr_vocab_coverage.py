"""Measure how much of a MEDS dataset the clmbr-t-base vocabulary covers.

femr 0.2.3 silently drops measurements its flat tokenizer can't map to a token (see FEMRTokenizer.get_feature_codes),
so low coverage means the model sees little of each patient's record, without any error.

Usage: python tools/clmbr_vocab_coverage.py PATH_TO_MEDS_DATASET [--max-files N]
"""

import argparse
import glob
import os

import polars as pl


def classify(lf: pl.LazyFrame, code_lookup, numeric_lookup, string_lookup) -> pl.LazyFrame:
    """Returns (code, kind, covered) per measurement, mirroring flat FEMRTokenizer.get_feature_codes."""
    if "text_value" not in lf.collect_schema().names():
        lf = lf.with_columns(pl.lit(None, dtype=pl.String).alias("text_value"))
    lf = lf.select(
        pl.col("code").cast(pl.String), pl.col("numeric_value").cast(pl.Float64), pl.col("text_value").cast(pl.String)
    ).with_row_index("row")

    codes = pl.LazyFrame({"code": list(code_lookup)}, schema={"code": pl.String})
    bins = pl.LazyFrame(
        [(code, start, end) for code, entries in numeric_lookup.items() for start, end, _ in entries],
        schema={"code": pl.String, "start": pl.Float64, "end": pl.Float64},
        orient="row",
    )
    texts = pl.LazyFrame(list(string_lookup), schema={"code": pl.String, "text_value": pl.String}, orient="row")
    covered = pl.lit(True).alias("covered")

    # Numeric measurements only map to one of their code's value bins, never to the bare code
    numeric = lf.filter(pl.col("numeric_value").is_not_null())
    in_bin = (
        numeric.join(bins, on="code")
        .filter(pl.col("numeric_value").is_between(pl.col("start"), pl.col("end"), closed="left"))
        .select("row")
        .unique()
        .with_columns(covered)
    )
    numeric = numeric.join(in_bin, on="row", how="left").with_columns(kind=pl.lit("numeric"))

    text = lf.filter(pl.col("numeric_value").is_null() & pl.col("text_value").is_not_null())
    text = text.join(texts.with_columns(covered), on=["code", "text_value"], how="left").with_columns(
        kind=pl.lit("text")
    )

    plain = lf.filter(pl.col("numeric_value").is_null() & pl.col("text_value").is_null())
    plain = plain.join(codes.with_columns(covered), on="code", how="left").with_columns(kind=pl.lit("code"))

    return pl.concat([f.select("code", "kind", pl.col("covered").fill_null(False)) for f in (numeric, text, plain)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meds_dataset", help="MEDS dataset root, the directory containing data/")
    parser.add_argument("--model", default="StanfordShahLab/clmbr-t-base")
    parser.add_argument("--max-files", type=int, help="Only read the first N data files")
    args = parser.parse_args()

    import femr.models.tokenizer

    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(args.model)
    assert not tokenizer.is_hierarchical, "This script mirrors the flat tokenizer only"

    files = sorted(glob.glob(os.path.join(args.meds_dataset, "data", "**", "*.parquet"), recursive=True))
    files = files[: args.max_files]
    assert files, f"No parquet files under {os.path.join(args.meds_dataset, 'data')}"

    measurements = classify(
        pl.scan_parquet(files), tokenizer.code_lookup, tokenizer.numeric_lookup, tokenizer.string_lookup
    )
    prefix = pl.col("code").str.split("/").list.first()
    coverage = [pl.len().alias("rows"), pl.col("covered").mean().alias("covered")]
    by_kind, by_vocabulary, uncovered = pl.collect_all(
        [
            measurements.group_by("kind").agg(coverage).sort("rows", descending=True),
            measurements.group_by(prefix.alias("vocabulary")).agg(coverage).sort("rows", descending=True).head(30),
            measurements.filter(~pl.col("covered"))
            .group_by("code", "kind")
            .len()
            .sort("len", descending=True)
            .head(30),
        ]
    )

    model_codes = (
        set(tokenizer.code_lookup) | set(tokenizer.numeric_lookup) | {code for code, _ in tokenizer.string_lookup}
    )
    model_vocabularies = (
        pl.DataFrame({"code": sorted(model_codes)})
        .group_by(prefix.alias("vocabulary"))
        .len()
        .sort("len", descending=True)
    )

    total = by_kind["rows"].sum()
    overall = (by_kind["rows"] * by_kind["covered"]).sum() / total
    with pl.Config(tbl_rows=30, fmt_str_lengths=60):
        print(f"{len(files)} files, {total:,} measurements, {overall:.1%} map to a model token\n")
        print("By measurement kind:", by_kind, sep="\n")
        print("\nBy vocabulary (code prefix) in your data:", by_vocabulary, sep="\n")
        print("\nMost frequent unmapped codes:", uncovered, sep="\n")
        print("\nVocabularies in the model:", model_vocabularies.head(30), sep="\n")


if __name__ == "__main__":
    main()
