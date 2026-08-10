"""Unit tests for the preprocessing, metric and ensembling components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from gbc.dataset import ROIDataset, class_weights
from gbc.ensemble import evaluate_combiners, fit_weights, greedy_selection
from gbc.metrics import (
    bootstrap_ci,
    compute_metrics,
    expected_calibration_error,
    mcnemar_test,
    temperature_scale,
)
from gbc.transforms import LoadROI, apply_clahe, build_transforms


def test_letterbox_preserves_aspect_ratio_and_pads():
    tall = Image.new("L", (40, 200), color=128)
    out = LoadROI(224, clahe=False)(tall)
    assert out.size == (224, 224)
    pixels = np.asarray(out.convert("L"))
    # the content column is centred; the left and right margins are padding
    assert pixels[:, 0].max() == 0 and pixels[:, -1].max() == 0
    assert pixels[112, 112] > 0


def test_clahe_increases_contrast_of_a_flat_gradient():
    ramp = np.tile(np.linspace(100, 140, 64, dtype=np.uint8), (64, 1))
    before = np.asarray(Image.fromarray(ramp), dtype=float)
    after = np.asarray(apply_clahe(Image.fromarray(ramp)), dtype=float)
    assert after.std() > before.std()
    assert after.min() >= 0 and after.max() <= 255


def test_eval_transform_is_deterministic_and_train_is_not():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (90, 120), dtype=np.uint8))
    ev = build_transforms(224, train=False)
    assert torch.allclose(ev(image), ev(image))
    torch.manual_seed(0)
    tr = build_transforms(224, train=True)
    assert not torch.allclose(tr(image), tr(image))


def test_dataset_targets_match_the_task():
    frame = pd.DataFrame([{
        "roi_path": "data/training/nml/" + __import__("os").listdir("data/training/nml")[0],
        "diagnosis_idx": 0, "stone": 1, "mural_thickening": 0, "malignancy": 1, "image_id": "x",
    }])
    tf = build_transforms(64, train=False, clahe=False)
    assert ROIDataset(frame, tf, "diagnosis")[0][1].item() == 0
    assert torch.equal(ROIDataset(frame, tf, "pathology")[0][1], torch.tensor([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        ROIDataset(frame, tf, "nonsense")


def test_class_weights_are_inverse_frequency():
    frame = pd.DataFrame({"diagnosis_idx": [0] * 80 + [1] * 10 + [2] * 10})
    w = class_weights(frame)
    assert w[0] < w[1] and pytest.approx(float(w.mean()), abs=1e-6) == 1.0


def test_metrics_on_a_perfect_classifier():
    y = np.array([0, 1, 2, 2, 1, 0])
    probs = np.eye(3)[y]
    m = compute_metrics(probs, y)
    assert m.accuracy == 1.0 and m.balanced_accuracy == 1.0
    assert m.malignant_sensitivity == 1.0 and m.malignant_specificity == 1.0


def test_malignant_sensitivity_counts_the_right_cell():
    y = np.array([2, 2, 2, 0])
    probs = np.eye(3)[[2, 0, 0, 0]]  # 1 of 3 malignant caught, no false positives
    m = compute_metrics(probs, y)
    assert pytest.approx(m.malignant_sensitivity) == 1 / 3
    assert m.malignant_specificity == 1.0


def test_ece_is_zero_for_a_calibrated_and_high_for_an_overconfident_model():
    y = np.array([0, 1] * 50)
    calibrated = np.tile([0.5, 0.5, 0.0], (100, 1))
    assert expected_calibration_error(calibrated, y) < 0.05
    overconfident = np.eye(3)[np.zeros(100, dtype=int)] * 0.99 + 0.005
    assert expected_calibration_error(overconfident, y) > 0.4


def test_temperature_scaling_cools_an_overconfident_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    logits = rng.normal(size=(400, 3))
    logits[np.arange(400), y] += 1.0
    assert temperature_scale(logits * 8, y) > 1.0  # inflated logits need T > 1


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    logits = rng.normal(size=(200, 3))
    logits[np.arange(200), y] += 2.0
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    point = compute_metrics(probs, y).accuracy
    lo, hi = bootstrap_ci(probs, y, n_resamples=200)["accuracy"]
    assert lo <= point <= hi


def test_mcnemar_is_symmetric_and_null_when_predictions_agree():
    y = np.array([0, 1, 2, 0, 1])
    assert mcnemar_test(y, y, y)["p_value"] == 1.0
    a, b = np.array([0, 1, 2, 0, 1]), np.array([1, 1, 2, 0, 1])
    assert mcnemar_test(y, a, b)["a_correct_b_wrong"] == mcnemar_test(y, b, a)["b_correct_a_wrong"]


def _toy_ensemble(n=180, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    meta = pd.DataFrame({"diagnosis_idx": y, "fold": np.arange(n) % 3})
    probs = {}
    for i, strength in enumerate([2.5, 1.5, 0.4]):
        logits = rng.normal(size=(n, 3))
        logits[np.arange(n), y] += strength
        e = np.exp(logits)
        probs[f"m{i}"] = e / e.sum(1, keepdims=True)
    return probs, meta, y


def test_fitted_weights_form_a_simplex_and_favour_the_stronger_member():
    probs, meta, y = _toy_ensemble()
    w = fit_weights(probs, y, np.arange(len(y)))
    assert pytest.approx(float(w.sum())) == 1.0 and (w >= 0).all()
    assert w[0] > w[2]


def test_combiners_are_scored_out_of_fold_and_beat_the_weakest_member():
    from sklearn.metrics import balanced_accuracy_score

    probs, meta, y = _toy_ensemble()
    combined, table = evaluate_combiners(probs, meta)
    assert set(combined) == {"mean", "weighted", "stack"}
    for p in combined.values():
        assert p.shape == (len(y), 3)
        assert np.allclose(p.sum(1), 1.0)
        assert balanced_accuracy_score(y, p.argmax(1)) > balanced_accuracy_score(y, probs["m2"].argmax(1))
    assert list(table["member"])[0] == "m0"


def test_greedy_selection_picks_the_strongest_member_first():
    probs, meta, _ = _toy_ensemble()
    assert greedy_selection(probs, meta)[0] == "m0"
