#!/usr/bin/env python3
"""
Cirro preprocess script for multitme pipeline.

This script prepares the samplesheet.csv input file required by the
multitme Nextflow pipeline.

The pipeline expects a samplesheet with three columns: sample, scrna, xenium

Inputs are provided as two Cirro datasets:
- A single-cell h5ad dataset (scRNA) — appears in ds.files
- A Xenium / SpatialData dataset — appears only in ds.metadata['inputs']

The script identifies which input is scRNA by matching dataset IDs from
ds.files, then pairs each Xenium input with the corresponding scRNA file.
"""

from __future__ import annotations

import re

import pandas as pd
from cirro.helpers.preprocess_dataset import PreprocessDataset

SAMPLESHEET_REQUIRED_COLUMNS = ("sample", "scrna", "xenium")


def sanitize_sample_name(name: str) -> str:
    """Replace whitespace and other illegal characters with underscores."""
    return re.sub(r"[^\w\-.]", "_", name.strip())


def split_inputs(ds: PreprocessDataset) -> tuple[list[dict], list[dict]]:
    """
    Split ds.metadata['inputs'] into scRNA inputs and Xenium inputs.

    scRNA inputs are those whose dataset ID (extracted from dataPath) appears
    in the 'dataset' column of ds.files.  Everything else is treated as Xenium.

    Returns (scrna_inputs, xenium_inputs).
    """
    inputs = ds.metadata.get("inputs", [])
    ds.logger.info(f"Total inputs in ds.metadata['inputs']: {len(inputs)}")
    for i, inp in enumerate(inputs):
        ds.logger.info(
            f"  Input [{i}]: name='{inp.get('name', '')}', dataPath='{inp.get('dataPath', '')}'"
        )

    files = ds.files
    # Collect dataset IDs that have files listed in ds.files (these are scRNA datasets)
    file_dataset_ids: set[str] = set()
    if not files.empty and "dataset" in files.columns:
        file_dataset_ids = set(files["dataset"].dropna().unique())
    ds.logger.info(f"Dataset IDs present in ds.files: {file_dataset_ids}")

    scrna_inputs, xenium_inputs = [], []
    for inp in inputs:
        data_path = inp.get("dataPath", "")
        # Check if any known file dataset ID appears in this input's dataPath
        is_scrna = any(did in data_path for did in file_dataset_ids)
        tag = "scrna" if is_scrna else "xenium"
        ds.logger.info(f"  Classified '{inp.get('name', '')}' ({data_path}) → {tag}")
        (scrna_inputs if is_scrna else xenium_inputs).append(inp)

    ds.logger.info(
        f"Split result: {len(scrna_inputs)} scRNA input(s), {len(xenium_inputs)} Xenium input(s)."
    )
    return scrna_inputs, xenium_inputs


def scrna_file_for_dataset(dataset_id: str, ds: PreprocessDataset) -> str:
    """
    Return the actual file path for a scRNA dataset from ds.files.

    Falls back to the dataPath directory if no file is found.
    """
    files = ds.files
    if files.empty or "dataset" not in files.columns:
        return ""
    matches = files[files["dataset"] == dataset_id]
    if matches.empty:
        ds.logger.warning(f"  No files found in ds.files for dataset ID '{dataset_id}'.")
        return ""
    if len(matches) > 1:
        ds.logger.warning(
            f"  Multiple files for dataset '{dataset_id}', using first: {matches['file'].tolist()}"
        )
    path = matches.iloc[0]["file"]
    ds.logger.info(f"  Resolved scRNA file for dataset '{dataset_id}': {path}")
    return path


def build_samplesheet(ds: PreprocessDataset) -> pd.DataFrame:
    """
    Pair scRNA and Xenium inputs into samplesheet rows.

    Pairing is positional: 1st scRNA with 1st Xenium, etc.
    If counts differ, a warning is logged and pairing stops at the shorter list.
    Sample names are taken from the Xenium input name and sanitized.
    """
    scrna_inputs, xenium_inputs = split_inputs(ds)

    if not scrna_inputs:
        raise ValueError(
            "Could not identify any scRNA input. "
            "Ensure a single-cell h5ad dataset is selected as an input."
        )
    if not xenium_inputs:
        raise ValueError(
            "Could not identify any Xenium input. "
            "Ensure a Xenium/SpatialData dataset is selected as an input."
        )
    if len(scrna_inputs) != len(xenium_inputs):
        ds.logger.warning(
            f"Unequal number of scRNA ({len(scrna_inputs)}) and Xenium ({len(xenium_inputs)}) inputs. "
            f"Pairing by position up to min({len(scrna_inputs)}, {len(xenium_inputs)})."
        )

    rows = []
    for i, (scrna_inp, xenium_inp) in enumerate(zip(scrna_inputs, xenium_inputs, strict=False)):
        # Derive the actual scRNA file path from ds.files
        scrna_data_path = scrna_inp.get("dataPath", "")
        # Extract dataset ID: the UUID portion of the dataPath
        dataset_id_match = re.search(
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            scrna_data_path,
        )
        scrna_dataset_id = dataset_id_match.group(1) if dataset_id_match else ""
        ds.logger.info(f"  Pair [{i}]: scRNA dataset ID extracted as '{scrna_dataset_id}'")

        scrna_path = (
            scrna_file_for_dataset(scrna_dataset_id, ds) if scrna_dataset_id else scrna_data_path
        )
        if not scrna_path:
            ds.logger.warning(
                f"  Pair [{i}]: falling back to scRNA dataPath directory: {scrna_data_path}"
            )
            scrna_path = scrna_data_path

        xenium_path = xenium_inp.get("dataPath", "")
        raw_name = xenium_inp.get("name", f"sample_{i}")
        sample_name = sanitize_sample_name(raw_name)

        if sample_name != raw_name:
            ds.logger.info(f"  Pair [{i}]: sample name sanitized '{raw_name}' → '{sample_name}'")

        ds.logger.info(
            f"  Pair [{i}]: sample='{sample_name}', scrna='{scrna_path}', xenium='{xenium_path}'"
        )
        rows.append({"sample": sample_name, "scrna": scrna_path, "xenium": xenium_path})

    return pd.DataFrame(rows)


def prepare_samplesheet(ds: PreprocessDataset) -> pd.DataFrame:
    """Build, validate, write samplesheet.csv, and update pipeline params."""
    ds.logger.info("=== prepare_samplesheet: start ===")
    ds.logger.info(f"Current params: {ds.params}")

    samplesheet = build_samplesheet(ds)

    # Validate required columns
    for colname in SAMPLESHEET_REQUIRED_COLUMNS:
        if colname not in samplesheet.columns:
            ds.logger.warning(
                f"Samplesheet missing column '{colname}' — populating with empty string."
            )
            samplesheet[colname] = ""

    ds.logger.info(f"Final samplesheet ({len(samplesheet)} row(s)):\n{samplesheet.to_string()}")
    samplesheet.to_csv("samplesheet.csv", index=False)
    ds.logger.info("Written to samplesheet.csv.")

    # Remove any samplesheet column keys from params to avoid conflicts
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
    if not ds.files.empty:
        ds.logger.info(f"ds.files content:\n{ds.files.to_string()}")
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
