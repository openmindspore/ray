import re

import pytest

from ray.air.constants import MAX_REPR_LENGTH
from ray.train.mindspore import MindSporeCheckpoint

@pytest.mark.parametrize(
    "checkpoint",
    [
        MindSporeCheckpoint(data_dict={"foo": "bar"}),
    ],
)
def test_repr(checkpoint):
    representation = repr(checkpoint)

    assert len(representation) < MAX_REPR_LENGTH
    pattern = re.compile(f"^{checkpoint.__class__.__name__}\\((.*)\\)$")
    assert pattern.match(representation)

if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", "-x", __file__]))
