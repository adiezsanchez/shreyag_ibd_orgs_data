import os
from pathlib import Path

import polars as pl


def _ensure_directory_exists(directory_path):
    """
    Ensure that a directory path exists.

    Parameters
    ----------
    directory_path : str | os.PathLike
        Path to the directory that should exist.

    Returns
    -------
    None
        This function creates the directory when missing and returns nothing.
    """
    os.makedirs(directory_path, exist_ok=True)


def merge_csv_files(
    results_directory: "Path",
    df_conditions: "pl.DataFrame",
) -> None:
    """
    Merge all CSV files in a directory and save one parquet dataset.

    The function joins per-file measurements with condition metadata and writes
    a single parquet file under ``./processed_data``. If the output file
    already exists, processing is skipped.

    Parameters
    ----------
    results_directory : Path
        Path to the directory containing CSV files to be merged.
    df_conditions : pl.DataFrame
        Condition metadata containing at least the ``well_id`` column.

    Returns
    -------
    None
        The merged data is written to
        ``./processed_data/{experiment_id}.parquet``.

    Raises
    ------
    ValueError
        If the input directory contains no CSV files.
    """

    # Extract the experiment name from the results directory
    experiment_id = results_directory.name

    # Construct .parquet savepath to check if it has been precomputed
    processed_data_dir = "./processed_data"
    _ensure_directory_exists(processed_data_dir)
    output_path = os.path.join(processed_data_dir, f"{experiment_id}.parquet")

    # Check if output parquet file already exists, skip if it does
    if os.path.exists(output_path):
        print(f"Output Parquet {output_path} already exists, skipping merge for {experiment_id}.")
        return None

    # Get all csv files
    csv_files = sorted(results_directory.glob("*.csv"))

    if not csv_files:
        raise ValueError("No CSV files found in folder")

    # Read and concatenate all CSVs using polars
    dfs = [pl.read_csv(str(f)) for f in csv_files]
    df = pl.concat(dfs, how="vertical_relaxed")

    # Merge with condition metadata (left join on 'well_id')
    df_merged = df.join(df_conditions, on="well_id", how="left")

    # Sanity check: Wells in df without condition info
    missing = df_merged["condition"].is_null().sum()
    print(f"Rows without condition: {missing}")

    # Sanity check: unique wells before/after
    unique_wells_before = df["well_id"].n_unique()
    unique_wells_after = df_merged["well_id"].n_unique()
    print(
        f"Unique wells before: {unique_wells_before}",
        f"Unique wells after: {unique_wells_after}",
    )

    # Save the merged dataframe to ./processed_data/ as {experiment_id}.parquet
    df_merged.write_parquet(output_path)
    print(f"Saved merged dataframe to {output_path}")

    return None


def get_unique_values(df, column_name):
    """
    Return sorted unique non-null values from a column as strings.

    Parameters
    ----------
    df : pl.LazyFrame | pl.DataFrame
        Input dataframe containing the requested column.
    column_name : str
        Name of the column from which unique values are extracted.

    Returns
    -------
    list[str]
        Sorted unique values represented as strings.
    """
    options = sorted(
        str(d)
        for d in set(df.select(column_name).collect().get_column(column_name).to_list())
        if d is not None
    )
    return options


def build_condition_order_by_group(df: pl.DataFrame | pl.LazyFrame) -> dict[str, list[str]]:
    """
    Map each group_number to treatment (condition) names in plate layout order.

    Order is derived by sorting wells within each group so legend colors stay
    consistent when changing donors, without forcing every treatment globally.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Dataframe containing ``group_number``, ``well_id``, and ``condition``.

    Returns
    -------
    dict[str, list[str]]
        Mapping from group identifier to ordered condition labels.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    order_by_group: dict[str, list[str]] = {}
    group_numbers = [
        str(g)
        for g in df.select("group_number")
        .unique()
        .sort("group_number")
        .get_column("group_number")
        .to_list()
        if g is not None
    ]

    for group in group_numbers:
        conditions = (
            df.filter(pl.col("group_number").cast(pl.Utf8) == group)
            .select(["well_id", "condition"])
            .unique()
            .sort("well_id")
            .get_column("condition")
            .to_list()
        )
        seen: set[str] = set()
        ordered: list[str] = []
        for condition in conditions:
            if condition is None:
                continue
            label = str(condition)
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        order_by_group[group] = ordered

    return order_by_group


def condition_hue_order(
    condition_order_by_group: dict[str, list[str]],
    *,
    selected_group: str | None,
    present_conditions: set[str],
) -> list[str] | None:
    """
    Build a seaborn hue order constrained to conditions present in data.

    Parameters
    ----------
    condition_order_by_group : dict[str, list[str]]
        Mapping of group identifier to ordered condition labels.
    selected_group : str | None
        Currently selected group. Use ``None`` or ``"None"`` for all groups.
    present_conditions : set[str]
        Condition labels present in the current filtered dataset.

    Returns
    -------
    list[str] | None
        Ordered condition labels for plotting, or ``None`` when no labels
        are available.
    """
    if selected_group and selected_group != "None":
        group_order = condition_order_by_group.get(selected_group, [])
        return [c for c in group_order if c in present_conditions]

    hue_order: list[str] = []
    seen: set[str] = set()
    for group in sorted(condition_order_by_group):
        for condition in condition_order_by_group[group]:
            if condition in present_conditions and condition not in seen:
                hue_order.append(condition)
                seen.add(condition)
    for condition in sorted(present_conditions - seen):
        hue_order.append(condition)
    return hue_order or None
