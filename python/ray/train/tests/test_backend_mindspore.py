import pytest

import ray

from ray.train._internal.backend_executor import (
    BackendExecutor,
)
from ray.train._internal.dataset_spec import RayDatasetSpec

from ray.train.mindspore import MindSporeConfig

@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()

EMPTY_RAY_DATASET_SPEC = RayDatasetSpec(dataset_or_dict=None)

def test_mindspore_start(ray_start_2_cpus):
    num_workers = 2
    mindspore_config = MindSporeConfig()
    e = BackendExecutor(mindspore_config, num_workers=num_workers)
    e.start()
    
    """Refer to https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.communication.html """
    def check_process_group():
        import mindspore
        from mindspore.communication.management import init, get_group_size

        return (
            mindspore.get_group_size() == 2
        )

    e.start_training(check_process_group, dataset_spec=EMPTY_RAY_DATASET_SPEC)
    assert all(e.finish_training())

    e._backend.on_shutdown(e.worker_group, e._backend_config)

    e.start_training(check_process_group, dataset_spec=EMPTY_RAY_DATASET_SPEC)
    assert not any(e.finish_training())

