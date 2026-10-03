"""Prepare a grouped tabular split for earthquake damage classification.

The source provides labeled building rows and an unlabeled example test file.
This preparer uses the example test file only for schema validation, creates a
fresh stratified holdout from labeled rows, groups exact non-ID feature
duplicates together, remaps row identifiers, and publishes coarse building
descriptors so held-out rows cannot be matched back to exact source records.
"""

from pathlib import Path
import hashlib
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from mlebench.utils import read_csv

ID_COL = "building_id"
TARGET_COL = "damage_grade"
RANDOM_STATE = 20260622
N_SPLITS = 10
LABELS = {1, 2, 3}
MIN_SOURCE_MATCHES_PER_PUBLIC_FEATURE_VECTOR = 5
ID_HASH_KEY = b"ml-olympiad-earthquake-damage-leakage-safe-ids-v2"
MIN_PUBLIC_ID = 1_000_000_000_000
PUBLIC_ID_SPACE = 8_000_000_000_000


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _feature_group_keys(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    separator = "\x1f"
    as_text = frame[feature_columns].astype(str)
    assert not as_text.apply(lambda col: col.str.contains(separator, regex=False).any()).any()
    return as_text.agg(separator.join, axis=1)


def _select_holdout(train: pd.DataFrame, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    overall = train[TARGET_COL].value_counts(normalize=True).sort_index()
    target_size = len(train) / N_SPLITS

    best_score: float | None = None
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    for train_idx, test_idx in splitter.split(train, train[TARGET_COL], groups):
        heldout = train.iloc[test_idx]
        proportions = heldout[TARGET_COL].value_counts(normalize=True).reindex(
            overall.index, fill_value=0.0
        )
        score = abs(len(test_idx) - target_size) / target_size
        score += float((proportions - overall).abs().sum())
        if best_score is None or score < best_score:
            best_score = score
            best_split = (np.sort(train_idx), np.sort(test_idx))

    assert best_split is not None, "No split was produced"
    return best_split


def _public_id_for_source_id(source_id: int, used: set[int]) -> int:
    counter = 0
    while True:
        message = f"{int(source_id)}:{counter}".encode("ascii")
        digest = hashlib.blake2b(message, digest_size=8, key=ID_HASH_KEY).digest()
        candidate = MIN_PUBLIC_ID + (
            int.from_bytes(digest, byteorder="big") % PUBLIC_ID_SPACE
        )
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def _assign_public_ids(frame: pd.DataFrame) -> pd.Series:
    used: set[int] = set()
    public_ids = [
        _public_id_for_source_id(source_id, used) for source_id in frame[ID_COL].to_numpy()
    ]
    return pd.Series(public_ids, index=frame.index, name=ID_COL, dtype=np.int64)


def _derive_public_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features[ID_COL] = frame[ID_COL].to_numpy()

    features["floor_count_group"] = np.where(
        frame["count_floors_pre_eq"].eq(1), "one", "two_or_more"
    )
    features["age_group"] = np.where(frame["age"].eq(0), "new_build", "older")
    features["area_group"] = np.where(
        frame["area_percentage"].le(8), "small_medium", "large"
    )
    features["height_group"] = np.where(
        frame["height_percentage"].le(5), "low_mid", "tall"
    )
    features["foundation_group"] = (
        frame["foundation_type"]
        .map(
            {
                "i": "engineered",
                "u": "engineered",
                "h": "cement_stone",
                "r": "masonry",
                "w": "other",
            }
        )
        .fillna("other")
    )

    return features


def _assert_public_features_are_not_source_fingerprints(
    public_features: pd.DataFrame, source_features: pd.DataFrame, source_labels: pd.Series
) -> None:
    feature_columns = [column for column in public_features.columns if column != ID_COL]
    source_keys = _feature_group_keys(source_features, feature_columns)
    public_keys = _feature_group_keys(public_features, feature_columns)
    source_key_summary = (
        pd.DataFrame({"key": source_keys, TARGET_COL: source_labels})
        .groupby("key")[TARGET_COL]
        .agg(size="size", label_count="nunique")
    )

    matched = source_key_summary.loc[public_keys]
    assert matched["size"].min() >= MIN_SOURCE_MATCHES_PER_PUBLIC_FEATURE_VECTOR, (
        "Public feature vectors must not identify a small number of source rows"
    )
    assert matched["label_count"].min() >= 2, (
        "Public feature vectors must not determine a single source label"
    )


def prepare(raw: Path, public: Path, private: Path) -> None:
    _reset_dir(public)
    _reset_dir(private)

    source = (
        read_csv(raw / "train.csv")
        .sort_values(ID_COL, kind="mergesort")
        .reset_index(drop=True)
    )
    raw_test = read_csv(raw / "test.csv")
    assert TARGET_COL in source.columns, "Source train file has no target column"
    assert TARGET_COL not in raw_test.columns, "Source test file unexpectedly contains labels"
    assert set(source[TARGET_COL].unique()) == LABELS, "Unexpected source labels"
    assert source[ID_COL].is_unique, "Source building IDs must be unique"

    feature_columns = [column for column in source.columns if column not in (ID_COL, TARGET_COL)]
    assert list(raw_test.columns) == [ID_COL] + feature_columns, "Unexpected source test schema"
    groups = _feature_group_keys(source, feature_columns)
    train_idx, test_idx = _select_holdout(source, groups)

    split_marker = pd.Series("", index=source.index)
    split_marker.iloc[train_idx] = "train"
    split_marker.iloc[test_idx] = "test"
    assert (split_marker != "").all(), "Every labeled row must appear in exactly one split"
    assert set(train_idx).isdisjoint(set(test_idx)), "Train and test splits overlap"

    source_public_features = _derive_public_features(source)
    _assert_public_features_are_not_source_fingerprints(
        source_public_features.iloc[test_idx],
        source_public_features,
        source[TARGET_COL],
    )

    public_id_map = _assign_public_ids(source)
    train_frame = source_public_features.iloc[train_idx].copy()
    test_frame = source_public_features.iloc[test_idx].copy()
    train_frame[TARGET_COL] = source.iloc[train_idx][TARGET_COL].to_numpy()
    test_frame[TARGET_COL] = source.iloc[test_idx][TARGET_COL].to_numpy()
    train_frame[ID_COL] = public_id_map.loc[train_frame.index].to_numpy()
    test_frame[ID_COL] = public_id_map.loc[test_frame.index].to_numpy()

    public_feature_columns = [
        column for column in train_frame.columns if column not in (ID_COL, TARGET_COL)
    ]
    train_public = train_frame.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    test_public = (
        test_frame.drop(columns=[TARGET_COL])
        .sort_values(ID_COL, kind="mergesort")
        .reset_index(drop=True)
    )
    answers = (
        test_frame[[ID_COL, TARGET_COL]]
        .sort_values(ID_COL, kind="mergesort")
        .reset_index(drop=True)
    )
    sample = answers[[ID_COL]].copy()
    sample[TARGET_COL] = 1

    assert train_public[ID_COL].is_unique and test_public[ID_COL].is_unique
    assert set(train_public[ID_COL]).isdisjoint(set(test_public[ID_COL]))
    all_public_ids = pd.concat([train_public[ID_COL], test_public[ID_COL]], ignore_index=True)
    assert all_public_ids.min() >= MIN_PUBLIC_ID
    assert all_public_ids.max() < MIN_PUBLIC_ID + PUBLIC_ID_SPACE
    assert set(all_public_ids).isdisjoint(set(source[ID_COL])), "Public IDs must not reuse source IDs"
    sorted_ids = np.sort(all_public_ids.to_numpy())
    assert not np.array_equal(sorted_ids, np.arange(sorted_ids[0], sorted_ids[0] + len(sorted_ids)))
    assert TARGET_COL not in test_public.columns, "Public test data must not contain labels"
    assert list(sample.columns) == list(answers.columns)
    assert len(sample) == len(answers)
    assert sample[ID_COL].equals(answers[ID_COL])
    assert set(answers[TARGET_COL].unique()) == LABELS

    full_train_features = pd.MultiIndex.from_frame(source.iloc[train_idx][feature_columns])
    full_test_features = pd.MultiIndex.from_frame(source.iloc[test_idx][feature_columns])
    assert full_train_features.intersection(full_test_features).empty, (
        "Exact source feature rows must not cross the split"
    )
    _assert_public_features_are_not_source_fingerprints(
        test_public[[ID_COL] + public_feature_columns],
        source_public_features,
        source[TARGET_COL],
    )

    train_public.to_csv(public / "train.csv", index=False)
    test_public.to_csv(public / "test.csv", index=False)
    sample.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)

    assert (public / "train.csv").is_file()
    assert (public / "test.csv").is_file()
    assert (public / "sample_submission.csv").is_file()
    assert (private / "answers.csv").is_file()
