import pytest
from ray.air import session
from ray.air.checkpoint import Checkpoint
import mindspore

import ray
from ray.train.mindspore import MindSporePredictor, MindSporeTrainer
from ray.tune import TuneError
from ray.air.config import ScalingConfig
import ray.train as train
from unittest.mock import patch
from ray.cluster_utils import Cluster

@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


if __name__ == "__main__":
    import sys

    import pytest
    
    sys.exit(pytest.main(["-v", "-x", __file__]))