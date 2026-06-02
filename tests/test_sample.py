from pathlib import Path

import pandas as pd
import polars as pl

from src.utils_data_analysis import (
    build_condition_order_by_group,
    condition_hue_order,
    get_unique_values,
    merge_csv_files,
)
from src.utils_data_plotting import _to_pandas, build_histogram, plot_plate_view


def test_get_unique_values_returns_sorted_strings():
    df = pl.DataFrame({"donor_id": [2, 1, 2, None]})
    result = get_unique_values(df.lazy(), "donor_id")
    assert result == ["1", "2"]


def test_build_condition_order_by_group_preserves_well_order():
    df = pl.DataFrame(
        {
            "group_number": ["1", "1", "1", "2", "2"],
            "well_id": ["A02", "A01", "A02", "B02", "B01"],
            "condition": ["cond_b", "cond_a", "cond_b", "cond_d", "cond_c"],
        }
    )
    result = build_condition_order_by_group(df)
    assert result["1"] == ["cond_a", "cond_b"]
    assert result["2"] == ["cond_c", "cond_d"]


def test_condition_hue_order_group_specific_and_global():
    mapping = {
        "1": ["a", "b"],
        "2": ["c", "d"],
    }
    specific = condition_hue_order(mapping, selected_group="1", present_conditions={"a", "x"})
    global_order = condition_hue_order(mapping, selected_group="None", present_conditions={"d", "a"})
    assert specific == ["a"]
    assert global_order == ["a", "d"]


def test_to_pandas_accepts_polars_and_pandas():
    pldf = pl.DataFrame({"x": [1, 2]})
    pdf = pd.DataFrame({"x": [3, 4]})
    assert list(_to_pandas(pldf)["x"]) == [1, 2]
    assert list(_to_pandas(pdf)["x"]) == [3, 4]


def test_build_histogram_returns_figure():
    df = pl.DataFrame({"value": [1, 2, 2, 3], "group": ["a", "a", "b", "b"]})
    fig = build_histogram(df, x_var="value", hue_var="group")
    assert fig is not None


def test_plot_plate_view_returns_figure_without_display():
    df = pl.DataFrame(
        {
            "well_id": ["A01", "A02", "B01"],
            "score": [1.0, 2.0, 3.0],
        }
    )
    fig = plot_plate_view(
        df=df,
        column_name="score",
        title="test_plate",
        label="score",
        save_dir="unused",
        display=False,
    )
    assert fig is not None


def test_merge_csv_files_creates_parquet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiment_dir = tmp_path / "8040_CLDN1.OCLN.ECAD"
    experiment_dir.mkdir()

    csv_path = experiment_dir / "part1.csv"
    csv_path.write_text("well_id,value\nA01,1\nA02,2\n")

    conditions = pl.DataFrame(
        {
            "well_id": ["A01", "A02"],
            "condition": ["ctrl", "treated"],
            "group_number": ["1", "1"],
        }
    )

    merge_csv_files(Path(experiment_dir), conditions)
    output_path = tmp_path / "processed_data" / "8040_CLDN1.OCLN.ECAD.parquet"
    assert output_path.exists()
