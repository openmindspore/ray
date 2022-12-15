import logging
from dataclasses import dataclass

import ray
from ray.train.backend import BackendConfig, Backend
from ray.train._internal.worker_group import WorkerGroup
from ray.util import PublicAPI

logger = logging.getLogger(__name__)

@PublicAPI(stability="beta")
@dataclass
class MindSporeConfig(BackendConfig):
    @property
    def backend_cls(self):
        return _MindSporeBackend

class _MindSporeBackend(Backend):
    def on_start(self, worker_group: WorkerGroup, backend_config: MindSporeConfig):
        """TODO"""
        pass

if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", "-x", __file__]))