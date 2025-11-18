import os
import sys

import stim

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from surface_code.dem_utils import add_spatial_correlations_to_dem, remap_metadata_detectors


def test_add_spatial_correlations_noop_when_butterflies_present():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1")
    metadata = {
        "noise_model": {"p_x_error": 1e-3, "p_z_error": 1e-3},
        "mwpm_debug": {
            "spatial_pairs": {
                "Z": {"patch": [{"rows": (0, 1), "count": 1}]},
                "X": {},
            },
            "detector_context": {
                0: {"tag": "z_butterfly", "context": {"rows": (0, 1)}},
            },
        },
    }

    updated = add_spatial_correlations_to_dem(dem, metadata)

    assert str(updated) == str(dem)


def test_remap_metadata_detectors_preserves_measurement_context():
    metadata = {
        "mwpm_debug": {
            "detector_context": {
                0: {"tag": "z_temporal", "context": {"patch": "p", "row": 0}},
                "__measurements__": {5: {"patch": "p", "basis": "Z", "row": 0, "round": 0}},
            }
        }
    }

    remap_metadata_detectors(metadata, {0: 2})

    det_ctx = metadata["mwpm_debug"]["detector_context"]
    assert 2 in det_ctx and det_ctx[2]["tag"] == "z_temporal"
    assert "__measurements__" in det_ctx
    assert det_ctx["__measurements__"][5]["round"] == 0
