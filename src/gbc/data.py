"""Dataset reconstruction, leakage auditing and leakage-free splitting for GBCU.

The ``data/`` tree shipped with this repository is a *flattened* re-encoding of
the GBCU dataset (Basu et al., CVPR 2022): 2,294 region-of-interest (ROI) crops
extracted from 1,255 source ultrasound images of 218 patients.  File names carry
the provenance we need to undo the flattening::

    <row_id>_im<image_id>_<roi_kind>_<k>.png

``image_id`` identifies the *source* ultrasound image, and ``roi_kind`` names the
annotated box the crop was taken from:

====== ===========================================================
kind   meaning
====== ===========================================================
nml    gallbladder ROI of a normal study
abn    gallbladder ROI of an abnormal study (the *parent* box)
stn    stone box, nested inside the ``abn`` box
bmt    benign mural-thickening box, nested inside the ``abn`` box
malg   malignancy box, nested inside the ``abn`` box
====== ===========================================================

Because the sub-pathology boxes are nested inside the parent ``abn`` box, the
five ``roi_kind`` values are *not* mutually exclusive classes and must not be
used as the targets of a 5-way softmax.  :func:`build_index` recovers the two
well-posed tasks instead:

``diagnosis``
    The canonical GBCU 3-class label (normal / benign / malignant), one sample
    per source image, directly comparable to the published literature.
``pathology``
    A 3-way *multi-label* target (stone / mural thickening / malignancy) over
    the gallbladder ROI, which is what the nested boxes actually annotate.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

FILENAME_RE = re.compile(r"^(?P<row>\d+)_im(?P<image_id>\d+)_(?P<kind>[a-z]+)_(?P<k>\d+)\.png$")

ROI_KINDS = ("nml", "abn", "stn", "bmt", "malg")
PARENT_KINDS = ("nml", "abn")
PATHOLOGY_KINDS = ("stn", "bmt", "malg")

DIAGNOSIS_CLASSES = ("normal", "benign", "malignant")
PATHOLOGY_CLASSES = ("stone", "mural_thickening", "malignancy")

_PATHOLOGY_COLUMN = {"stn": "stone", "bmt": "mural_thickening", "malg": "malignancy"}

# Published composition of GBCU, used as an integrity check on the reconstruction.
GBCU_REFERENCE_COUNTS = {"normal": 432, "benign": 558, "malignant": 265}


@dataclass
class CropRecord:
    """One ROI crop on disk."""

    path: Path
    image_id: str
    kind: str
    original_split: str


def scan_crops(data_dir: str | Path) -> list[CropRecord]:
    """Walk ``data/{training,validation,test}/<kind>/`` and parse every crop."""
    data_dir = Path(data_dir)
    records: list[CropRecord] = []
    for split_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for kind_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for png in sorted(kind_dir.glob("*.png")):
                match = FILENAME_RE.match(png.name)
                if match is None:
                    raise ValueError(f"unparseable crop filename: {png}")
                if match["kind"] != kind_dir.name:
                    raise ValueError(f"kind mismatch between name and folder: {png}")
                records.append(
                    CropRecord(
                        path=png,
                        image_id=match["image_id"],
                        kind=match["kind"],
                        original_split=split_dir.name,
                    )
                )
    if not records:
        raise FileNotFoundError(f"no crops found under {data_dir}")
    return records


def build_index(data_dir: str | Path, *, verify: bool = True) -> pd.DataFrame:
    """Reconstruct one row per *source image* with both task targets.

    Returns a frame with columns ``image_id``, ``roi_path`` (the parent
    gallbladder ROI used as model input), ``diagnosis``, the three binary
    pathology columns, and the crop paths of each nested box.
    """
    by_image: dict[str, dict[str, CropRecord]] = defaultdict(dict)
    for record in scan_crops(data_dir):
        # A source image has at most one box of each kind in this release.
        by_image[record.image_id][record.kind] = record

    rows = []
    for image_id, kinds in sorted(by_image.items()):
        parents = [k for k in PARENT_KINDS if k in kinds]
        if len(parents) != 1:
            raise ValueError(
                f"image {image_id} has parent ROIs {parents}; expected exactly one of {PARENT_KINDS}"
            )
        parent = parents[0]
        pathologies = {k for k in PATHOLOGY_KINDS if k in kinds}

        if parent == "nml":
            if pathologies:
                raise ValueError(f"normal image {image_id} carries pathology boxes {pathologies}")
            diagnosis = "normal"
        elif "malg" in pathologies:
            diagnosis = "malignant"
        else:
            diagnosis = "benign"

        row = {
            "image_id": image_id,
            "parent_kind": parent,
            "roi_path": str(kinds[parent].path),
            "diagnosis": diagnosis,
            "diagnosis_idx": DIAGNOSIS_CLASSES.index(diagnosis),
            "n_pathology_boxes": len(pathologies),
            "original_splits": "|".join(sorted({r.original_split for r in kinds.values()})),
        }
        for kind, column in _PATHOLOGY_COLUMN.items():
            row[column] = int(kind in pathologies)
            row[f"{column}_path"] = str(kinds[kind].path) if kind in kinds else ""
        rows.append(row)

    index = pd.DataFrame(rows)
    if verify:
        counts = index["diagnosis"].value_counts().to_dict()
        if counts != GBCU_REFERENCE_COUNTS:
            raise ValueError(
                f"reconstruction does not match published GBCU counts: got {counts}, "
                f"expected {GBCU_REFERENCE_COUNTS}"
            )
    return index


@dataclass
class LeakageReport:
    """Quantifies how badly the shipped ``training/validation/test`` split leaks."""

    n_crops: int
    n_source_images: int
    crops_per_split: dict[str, int]
    images_in_multiple_splits: int
    train_test_shared_images: int
    train_val_shared_images: int
    val_test_shared_images: int
    contradictory_label_images: int
    nested_box_pairs: int
    nested_box_pairs_strictly_contained: int
    examples: list[dict] = field(default_factory=list)

    @property
    def leaked_fraction(self) -> float:
        return self.images_in_multiple_splits / self.n_source_images

    def to_markdown(self) -> str:
        lines = [
            "| quantity | value |",
            "| --- | --- |",
            f"| ROI crops on disk | {self.n_crops:,} |",
            f"| distinct source images | {self.n_source_images:,} |",
        ]
        for split, n in self.crops_per_split.items():
            lines.append(f"| crops in `{split}/` | {n:,} |")
        lines += [
            f"| source images appearing in >1 split | **{self.images_in_multiple_splits:,}** "
            f"({self.leaked_fraction:.1%}) |",
            f"| source images shared by train & test | **{self.train_test_shared_images:,}** |",
            f"| source images shared by train & val | {self.train_val_shared_images:,} |",
            f"| source images shared by val & test | {self.val_test_shared_images:,} |",
            f"| source images whose crops carry mutually contradictory 5-way labels "
            f"| {self.contradictory_label_images:,} |",
            f"| pathology/parent box pairs checked for nesting | {self.nested_box_pairs:,} |",
            f"| of those, pathology box strictly smaller than parent | "
            f"{self.nested_box_pairs_strictly_contained:,} "
            f"({self.nested_box_pairs_strictly_contained / max(self.nested_box_pairs, 1):.1%}) |",
        ]
        return "\n".join(lines)


def audit_leakage(data_dir: str | Path, *, n_examples: int = 5) -> LeakageReport:
    """Measure cross-split contamination in the shipped split."""
    from PIL import Image

    records = scan_crops(data_dir)
    splits_of: dict[str, set[str]] = defaultdict(set)
    kinds_of: dict[str, dict[str, CropRecord]] = defaultdict(dict)
    crops_per_split: dict[str, int] = defaultdict(int)
    for record in records:
        splits_of[record.image_id].add(record.original_split)
        kinds_of[record.image_id][record.kind] = record
        crops_per_split[record.original_split] += 1

    def shared(a: str, b: str) -> int:
        return sum(1 for s in splits_of.values() if a in s and b in s)

    n_pairs = n_contained = 0
    for kinds in kinds_of.values():
        parent = next((kinds[k] for k in PARENT_KINDS if k in kinds), None)
        if parent is None:
            continue
        parent_area = np.prod(Image.open(parent.path).size)
        for kind in PATHOLOGY_KINDS:
            if kind not in kinds:
                continue
            n_pairs += 1
            n_contained += int(np.prod(Image.open(kinds[kind].path).size) < parent_area)

    examples = [
        {
            "image_id": image_id,
            "crops": sorted(
                f"{r.original_split}/{r.kind}/{r.path.name}" for r in kinds_of[image_id].values()
            ),
        }
        for image_id in sorted(splits_of)
        if len(splits_of[image_id]) > 1
    ][:n_examples]

    return LeakageReport(
        n_crops=len(records),
        n_source_images=len(splits_of),
        crops_per_split=dict(sorted(crops_per_split.items())),
        images_in_multiple_splits=sum(1 for s in splits_of.values() if len(s) > 1),
        train_test_shared_images=shared("training", "test"),
        train_val_shared_images=shared("training", "validation"),
        val_test_shared_images=shared("validation", "test"),
        contradictory_label_images=sum(1 for k in kinds_of.values() if len(k) > 1),
        nested_box_pairs=n_pairs,
        nested_box_pairs_strictly_contained=n_contained,
        examples=examples,
    )


def assign_folds(
    index: pd.DataFrame, *, n_splits: int = 5, seed: int = 1337, group_col: str = "image_id"
) -> pd.DataFrame:
    """Add a ``fold`` column via stratified *grouped* k-fold.

    Grouping on ``image_id`` guarantees that every crop of a source image lands
    in one fold, which is the contamination the shipped split gets wrong.  See
    ``docs/data_leakage_audit.md`` for the residual patient-level caveat.
    """
    index = index.copy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    index["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        splitter.split(index, index["diagnosis_idx"], groups=index[group_col])
    ):
        index.iloc[val_idx, index.columns.get_loc("fold")] = fold
    assert (index["fold"] >= 0).all(), "every row must receive a fold"
    return index


def build_crop_index(data_dir: str | Path) -> pd.DataFrame:
    """One row per ROI *crop*, labelled with its source image's diagnosis.

    Used only for the leakage ablation in ``docs/data_leakage_audit.md``: it is
    the unit of analysis the shipped split operates on, so splitting these rows
    with and without grouping on ``image_id`` isolates the effect of source-image
    contamination while holding the data, model and schedule fixed.
    """
    image_index = build_index(data_dir).set_index("image_id")
    rows = []
    for record in scan_crops(data_dir):
        source = image_index.loc[record.image_id]
        rows.append({
            "image_id": record.image_id,
            "roi_path": str(record.path),
            "roi_kind": record.kind,
            "diagnosis": source["diagnosis"],
            "diagnosis_idx": int(source["diagnosis_idx"]),
            **{c: int(source[c]) for c in PATHOLOGY_CLASSES},
        })
    return pd.DataFrame(rows).sort_values(["image_id", "roi_kind"]).reset_index(drop=True)


def assign_folds_ungrouped(
    index: pd.DataFrame, *, n_splits: int = 5, seed: int = 1337
) -> pd.DataFrame:
    """Stratified k-fold that **ignores** ``image_id``, reproducing the leak."""
    from sklearn.model_selection import StratifiedKFold

    index = index.copy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    index["fold"] = -1
    for fold, (_, val_idx) in enumerate(splitter.split(index, index["diagnosis_idx"])):
        index.iloc[val_idx, index.columns.get_loc("fold")] = fold
    return index
