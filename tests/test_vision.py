from types import SimpleNamespace

import numpy as np
import pytest

from ar_lava.vision import PoseTracker


@pytest.mark.parametrize("shape", [(120, 160, 1), (60, 80, 1), (120, 160)])
def test_pose_mask_is_single_channel_and_matches_original_pixels(shape):
    point = SimpleNamespace(x=0.5, y=0.5, visibility=1, presence=1)
    mask = np.full(shape, 0.75, np.float32)
    result = SimpleNamespace(
        pose_landmarks=[[point] * 33],
        segmentation_masks=[SimpleNamespace(numpy_view=lambda: mask)],
    )
    tracker = PoseTracker.__new__(PoseTracker)
    tracker.last_ms = 0
    tracker.model = SimpleNamespace(detect_for_video=lambda image, stamp: result)
    _, output, people = tracker.process(np.zeros((120, 160, 3), np.uint8))
    assert people == 1 and output.shape == (120, 160)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, 0.75)
