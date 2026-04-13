#!/usr/bin/env python3
"""
Cirro preprocess script for multitme pipeline.

This script prepares the samplesheet.csv input file required by the
multitme Nextflow pipeline.

The pipeline accepts:
- scRNA-seq data (.h5ad)
- Xenium spatial data (.h5ad, Xenium Ranger output dir, or SpatialData zarr dir)

The samplesheet has three columns: sample, scrna, xenium

Input datasets can be provided in two ways:
1. Files in ds.files where scrna files contain 'scrna' and xenium files contain 'xenium'
2. Metadata inputs with explicit 'scrna' and 'xenium' paths per sample
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from cirro.helpers.preprocess_dataset import PreprocessDataset

SAMPLESHEET_REQUIRED_COLUMNS = (
    "sample",
    "scrna",
    "xenium",
)


def samplesheet_from_files(ds: PreprocessDataset) -> pd.DataFrame:
    """
    Create a samplesheet from Cirro's files DataFrame.

    Detects scrna vs xenium files by checking if 'scrna' or 'xenium' appears
    in the file path. Files not matching either pattern are skipped with a warning.
    """
    files = ds.files
    if files.empty:
        ds.logger.info("ds.files is empty — skipping file-based samplesheet construction.")
        return pd.DataFrame()

    ds.logger.info(
        f"Building samplesheet from {len(files)} file(s) across {files['sample'].nunique()} sample(s)."
    )
    ds.logger.info(f"All files:\n{files.to_string()}")

    rows = []
    samples = sorted(files["sample"].unique())
    ds.logger.info(f"Samples found: {samples}")

    for sample in samples:
        group = files[files["sample"] == sample]
        ds.logger.info(f"  Sample '{sample}': {len(group)} file(s) — {group['file'].tolist()}")

        scrna_files = group[group["file"].str.contains("scrna", case=False, na=False)]
        xenium_files = group[group["file"].str.contains("xenium", case=False, na=False)]
        unmatched = group[
            ~group["file"].str.contains("scrna", case=False, na=False)
            & ~group["file"].str.contains("xenium", case=False, na=False)
        ]

        if not unmatched.empty:
            ds.logger.warning(
                f"  Sample '{sample}': {len(unmatched)} file(s) didn't match 'scrna' or 'xenium' "
                f"and will be ignored: {unmatched['file'].tolist()}"
            )

        if scrna_files.empty:
            ds.logger.warning(f"  Sample '{sample}': no scrna file detected — skipping sample.")
            continue
        if xenium_files.empty:
            ds.logger.warning(f"  Sample '{sample}': no xenium file detected — skipping sample.")
            continue

        scrna_path = scrna_files.iloc[0]["file"]
        xenium_path = xenium_files.iloc[0]["file"]

        if len(scrna_files) > 1:
            ds.logger.warning(
                f"  Sample '{sample}': multiple scrna files found, using first: {scrna_path}. "
                f"Others ignored: {scrna_files.iloc[1:]['file'].tolist()}"
            )
        if len(xenium_files) > 1:
            ds.logger.warning(
                f"  Sample '{sample}': multiple xenium files found, using first: {xenium_path}. "
                f"Others ignored: {xenium_files.iloc[1:]['file'].tolist()}"
            )

        ds.logger.info(f"  Sample '{sample}': scrna={scrna_path}, xenium={xenium_path}")
        rows.append({"sample": sample, "scrna": scrna_path, "xenium": xenium_path})

    ds.logger.info(f"File-based samplesheet construction complete: {len(rows)} sample(s) matched.")
    return pd.DataFrame(rows)


def samplesheet_from_params(ds: PreprocessDataset) -> pd.DataFrame:
    """
    Create a samplesheet from dataset metadata inputs.

    Expects each input to have 'name', 'scrna', and 'xenium' keys,
    or falls back to using 'dataPath' for xenium if only one path is present.
    """
    inputs = ds.metadata.get("inputs", [])
    ds.logger.info(f"ds.metadata['inputs'] has {len(inputs)} entry/entries.")

    if not inputs:
        ds.logger.warning("No inputs found in ds.metadata — cannot build samplesheet from params.")
        return pd.DataFrame()

    rows = []
    for i, inp in enumerate(inputs):
        sample = inp.get("name", inp.get("sample", ""))
        scrna = inp.get("scrna", "")
        xenium = inp.get("xenium", inp.get("dataPath", ""))

        ds.logger.info(f"  Input [{i}]: sample='{sample}', scrna='{scrna}', xenium='{xenium}'")

        if not sample:
            ds.logger.warning(
                f"  Input [{i}]: missing 'name'/'sample' key — will appear as empty string."
            )
        if not scrna:
            ds.logger.warning(f"  Input [{i}]: missing 'scrna' key — will be empty.")
        if not xenium:
            ds.logger.warning(f"  Input [{i}]: missing 'xenium'/'dataPath' key — will be empty.")

        rows.append({"sample": sample, "scrna": scrna, "xenium": xenium})

    ds.logger.info(f"Params-based samplesheet construction complete: {len(rows)} row(s).")
    return pd.DataFrame(rows)


def prepare_samplesheet(ds: PreprocessDataset) -> pd.DataFrame:
    """
    Prepare the samplesheet for the pipeline.

    Tries to create from files first, falls back to params if no files found.
    Ensures all required columns are present and cleans up params.
    """
    ds.logger.info("=== prepare_samplesheet: start ===")
    ds.logger.info(f"Current params: {ds.params}")

    ds.logger.info("--- Attempting file-based samplesheet construction ---")
    samplesheet = samplesheet_from_files(ds)

    if samplesheet.empty:
        ds.logger.warning(
            "File-based construction yielded no rows. Falling back to params-based construction."
        )
        ds.logger.info("--- Attempting params-based samplesheet construction ---")
        samplesheet = samplesheet_from_params(ds)
        if samplesheet.empty:
            raise ValueError(
                "No files found in dataset and unable to prepare samplesheet from params."
            )
        ds.logger.info("Params-based construction succeeded.")
    else:
        ds.logger.info(f"File-based construction succeeded with {len(samplesheet)} sample(s).")

    # Ensure all required columns are present
    for colname in SAMPLESHEET_REQUIRED_COLUMNS:
        if colname not in samplesheet.columns:
            ds.logger.warning(
                f"Samplesheet is missing required column '{colname}'. Populating with NaN."
            )
            samplesheet[colname] = np.nan

    ds.logger.info(f"Final samplesheet ({len(samplesheet)} row(s)):\n{samplesheet.to_string()}")
    samplesheet.to_csv("samplesheet.csv", index=False)
    ds.logger.info("Written to samplesheet.csv.")

    # Remove samplesheet columns from params to avoid conflicts
    to_remove = [k for k in ds.params if k in SAMPLESHEET_REQUIRED_COLUMNS]
    if to_remove:
        ds.logger.info(f"Removing samplesheet columns from params to avoid conflicts: {to_remove}")
        for k in to_remove:
            ds.remove_param(k)
    else:
        ds.logger.info("No samplesheet column keys found in params — nothing to remove.")

    ds.add_param("input", "samplesheet.csv", overwrite=True)
    ds.logger.info("Set params.input = 'samplesheet.csv'.")

    ds.logger.info("=== prepare_samplesheet: done ===")
    return samplesheet


def main():
    ds = PreprocessDataset.from_running()

    ds.logger.info("========================================")
    ds.logger.info("multitme Cirro preprocess: start")
    ds.logger.info("========================================")
    ds.logger.info(f"ds.files: {len(ds.files)} row(s), columns: {list(ds.files.columns)}")
    ds.logger.info(f"ds.params: {ds.params}")
    ds.logger.info(f"ds.metadata keys: {list(ds.metadata.keys()) if ds.metadata else '(empty)'}")

    samplesheet = prepare_samplesheet(ds)

    ds.logger.info("========================================")
    ds.logger.info("multitme Cirro preprocess: complete")
    ds.logger.info(f"  Samplesheet rows : {len(samplesheet)}")
    ds.logger.info(f"  Final params     : {ds.params}")
    ds.logger.info("========================================")


if __name__ == "__main__":
    main()
