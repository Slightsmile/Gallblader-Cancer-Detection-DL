"""Invariants that protect the dataset reconstruction and the split protocol.

These are the claims `docs/data_leakage_audit.md` rests on; if any of them
breaks, the numbers in the paper are wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbc.data import (
    DIAGNOSIS_CLASSES,
    GBCU_REFERENCE_COUNTS,
    PATHOLOGY_CLASSES,
    assign_folds,
    assign_folds_ungrouped,
    audit_leakage,
    build_crop_index,
    build_index,
)

DATA = "data"


@pytest.fixture(scope="module")
def index() -> pd.DataFrame:
    return build_index(DATA)


def test_reconstruction_matches_published_gbcu(index):
    assert index["diagnosis"].value_counts().to_dict() == GBCU_REFERENCE_COUNTS
    assert len(index) == sum(GBCU_REFERENCE_COUNTS.values()) == 1255
    assert index["image_id"].is_unique


def test_label_hierarchy_is_consistent(index):
    normal = index[index["diagnosis"] == "normal"]
    assert (normal[list(PATHOLOGY_CLASSES)].sum(axis=1) == 0).all(), "normal studies carry no pathology box"
    assert (normal["parent_kind"] == "nml").all()

    abnormal = index[index["diagnosis"] != "normal"]
    assert (abnormal["parent_kind"] == "abn").all()
    assert (abnormal[list(PATHOLOGY_CLASSES)].sum(axis=1) >= 1).all(), "abnormal studies carry >=1 box"

    # malignancy is exactly the malignant class, and never co-occurs with mural thickening
    assert (index["malignancy"] == (index["diagnosis"] == "malignant")).all()
    assert not (index["malignancy"] & index["mural_thickening"]).any()


def test_shipped_split_leaks(index):
    report = audit_leakage(DATA)
    assert report.n_crops == 2294
    assert report.n_source_images == 1255
    assert report.train_test_shared_images > 0, "the documented leak must still be detectable"
    assert report.leaked_fraction > 0.3
    # The nested-ROI geometry is what makes the leak spatial rather than statistical.
    assert report.nested_box_pairs_strictly_contained / report.nested_box_pairs > 0.99


def test_grouped_folds_never_split_a_source_image(index):
    folded = assign_folds(index, n_splits=5, seed=1337)
    crops = build_crop_index(DATA).merge(folded[["image_id", "fold"]], on="image_id")
    assert crops.groupby("image_id")["fold"].nunique().eq(1).all()
    assert sorted(folded["fold"].unique()) == list(range(5))


def test_grouped_folds_are_stratified_and_balanced(index):
    folded = assign_folds(index, n_splits=5, seed=1337)
    sizes = folded.groupby("fold").size()
    assert sizes.max() - sizes.min() <= 5
    for cls in DIAGNOSIS_CLASSES:
        share = folded[folded["diagnosis"] == cls].groupby("fold").size() / sizes
        assert share.max() - share.min() < 0.03, f"{cls} is unevenly distributed across folds"


def test_folds_are_deterministic_given_a_seed(index):
    a = assign_folds(index, seed=7)["fold"].to_numpy()
    b = assign_folds(index, seed=7)["fold"].to_numpy()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, assign_folds(index, seed=8)["fold"].to_numpy())


def test_ungrouped_folds_do_split_source_images():
    """The ablation's control arm must actually reproduce the leak."""
    crops = assign_folds_ungrouped(build_crop_index(DATA), n_splits=3, seed=1337)
    multi_crop = crops.groupby("image_id").filter(lambda g: len(g) > 1)
    assert multi_crop.groupby("image_id")["fold"].nunique().gt(1).any()


def test_crop_index_covers_every_file_on_disk():
    crops = build_crop_index(DATA)
    assert len(crops) == 2294
    assert crops["roi_path"].is_unique
    assert set(crops["diagnosis"]) == set(DIAGNOSIS_CLASSES)
