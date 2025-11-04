"""Tests for DEM pruning of unused detectors."""

import os
import sys

import stim

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from surface_code.dem_utils import (  # noqa: E402
    prune_dem_to_live_detectors,
    collect_detectors_in_errors,
)


def build_dem(text: str) -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(text.strip())


def test_prune_drops_detectors_without_errors():
    dem = build_dem(
        """
        detector D0
        detector D1
        detector D2
        error(0.1) D0 D1
        # D2 never appears in any ERROR line
        """
    )
    metadata = {
        "mwpm_debug": {
            "detector_context": {
                0: {"tag": "z_temporal"},
                1: {"tag": "z_temporal"},
                2: {"tag": "z_temporal"},
            }
        },
        "boundary_anchors": {"detector_ids": [1, 2]},
    }
    new_dem, mapping = prune_dem_to_live_detectors(dem, metadata)

    assert new_dem.num_detectors == 2
    assert collect_detectors_in_errors(new_dem) == {0, 1}
    assert mapping == {0: 0, 1: 1}
    assert metadata["boundary_anchors"]["detector_ids"] == [1]
    ctx = metadata["mwpm_debug"]["detector_context"]
    assert set(ctx.keys()) == {0, 1}


def test_prune_keeps_detectors_with_singleton_errors():
    dem = build_dem(
        """
        detector D0
        error(0.01) D0
        """
    )
    new_dem, mapping = prune_dem_to_live_detectors(dem)
    assert new_dem.num_detectors == 1
    assert mapping == {0: 0}


def test_prune_respects_metadata_mapping():
    dem = build_dem(
        """
        detector D0
        detector D1
        detector D2
        error(0.05) D0 D1
        error(0.05) D0
        """
    )
    metadata = {
        "mwpm_debug": {
            "detector_context": {
                0: {"tag": "z_temporal"},
                1: {"tag": "x_temporal"},
                2: {"tag": "z_wrap"},
                10: {"tag": "orphan"},
            },
            "degree_violations": [2, 10],
            "odd_degree_details": {
                2: [{"tag": "orphan", "neighbor": 5}],
                10: [{"tag": "ghost", "neighbor": 3}],
            },
        },
        "boundary_anchors": {"detector_ids": [0, 2]},
    }
    new_dem, mapping = prune_dem_to_live_detectors(dem, metadata)

    assert new_dem.num_detectors == 2
    assert mapping == {0: 0, 1: 1}
    assert metadata["boundary_anchors"]["detector_ids"] == [0]
    ctx = metadata["mwpm_debug"]["detector_context"]
    assert set(ctx.keys()) == {0, 1}
    assert metadata["mwpm_debug"]["degree_violations"] == []
    assert metadata["mwpm_debug"]["odd_degree_details"] == {}
