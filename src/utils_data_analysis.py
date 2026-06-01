import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns


def plot_plate_view(df, column_name, title, label, save_dir, fmt=3, display=True, cmap="magma"):
    # --- Parse well_id into row (A–H) and column (1–12) ---
    def split_well_id(well):
        match = re.match(r"([A-H])(\d{1,2})", str(well))
        if match:
            row, col = match.groups()
            return row, int(col)
        return None, None

    df[["row", "col"]] = df["well_id"].apply(lambda x: pd.Series(split_well_id(x)))

    # --- Pivot into 96-well plate layout ---
    plate_matrix = df.pivot(index="row", columns="col", values=column_name)

    # Reindex rows and columns to enforce full plate structure
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    plate_matrix = plate_matrix.reindex(index=rows, columns=cols)

    # --- Plot heatmap ---
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        plate_matrix,
        cmap=cmap,  # or "coolwarm", "magma" etc.
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": label},
        annot=True,
        fmt=f".{fmt}f",
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Column")
    plt.ylabel("Row")

    # Rotate row (y-axis) labels 90° to the right
    ax.set_yticklabels(ax.get_yticklabels(), rotation=-90, va="center")

    # --- Save plot ---
    save_dir_full = f"{save_dir}/plate_view/{column_name}"
    os.makedirs(save_dir_full, exist_ok=True)
    save_path = os.path.join(save_dir_full, f"{title}_{column_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close()

    print(f"Saved plate view to {save_path}")


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
