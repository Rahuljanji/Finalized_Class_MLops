from pathlib import Path
import pandas as pd
import pytest

RAW_DATA_PATH = Path("data/raw/Iris.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")

EXPECTED_FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
EXPECTED_TARGET = "Species"
VALID_CLASSES = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


def test_raw_dataset_exists_and_valid():
    """Verify raw Iris dataset exists, has expected columns and no nulls."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)

    assert len(df) == 150, f"Expected 150 rows in raw Iris, got {len(df)}"
    for col in EXPECTED_FEATURES + [EXPECTED_TARGET]:
        assert col in df.columns, f"Missing column {col} in raw data"

    assert df[EXPECTED_FEATURES].isnull().sum().sum() == 0, "Null values found in features"
    assert set(df[EXPECTED_TARGET].unique()) == VALID_CLASSES


def test_preprocessed_splits_exist_and_consistent():
    """Verify train and test sets exist, have no duplicates, and match deduplicated total."""
    assert TRAIN_DATA_PATH.exists(), f"Train set not found at {TRAIN_DATA_PATH}"
    assert TEST_DATA_PATH.exists(), f"Test set not found at {TEST_DATA_PATH}"

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # 150 raw rows - 3 duplicate rows = 147 distinct rows
    total_rows = len(train_df) + len(test_df)
    assert total_rows == 147, f"Expected 147 deduplicated rows across splits, got {total_rows}"

    for col in EXPECTED_FEATURES + [EXPECTED_TARGET]:
        assert col in train_df.columns
        assert col in test_df.columns

    # Verify both splits contain all 3 classes (stratified)
    assert set(train_df[EXPECTED_TARGET].unique()) == VALID_CLASSES
    assert set(test_df[EXPECTED_TARGET].unique()) == VALID_CLASSES
