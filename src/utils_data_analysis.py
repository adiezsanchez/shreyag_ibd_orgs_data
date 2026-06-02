import os
from pathlib import Path

import polars as pl


def get_1st_99th_percentile(series):
    """
    Returns the 1st and 99th percentile values of a pandas Series as a tuple (min, max).
    """
    p1 = series.quantile(0.01)
    p99 = series.quantile(0.99)
    return (p1, p99)


def _ensure_directory_exists(directory_path):
    """
    Helper function to ensure a directory exists. Creates it if it does not.
    """
    os.makedirs(directory_path, exist_ok=True)


def merge_csv_files(
    results_directory: "Path",
    df_conditions: "pl.DataFrame",
) -> None:
    """
    Merge all CSV files in a directory, enrich with condition metadata,
    and save the result as a .parquet file using polars.

    Args
    ----
    results_directory : pathlib.Path
        Path to the directory containing CSV files to be merged.
    df_conditions : pl.DataFrame
        Polars DataFrame with condition metadata (must contain "well_id" column).

    Returns
    -------
    None
        The merged DataFrame is saved as a Parquet file
        in ./processed_data/{experiment_id}.parquet.
    """

    # Extract the experiment name from the results directory
    experiment_id = results_directory.name

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
    processed_data_dir = "./processed_data"
    _ensure_directory_exists(processed_data_dir)
    output_path = os.path.join(processed_data_dir, f"{experiment_id}.parquet")
    df_merged.write_parquet(output_path)
    print(f"Saved merged dataframe to {output_path}")

    return None


def get_unique_values(df, column_name):
    options = sorted(
        str(d)
        for d in set(df.select(column_name).collect().get_column(column_name).to_list())
        if d is not None
    )
    return options
