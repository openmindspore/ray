import time
import pytest

import ray
import ray.train as train
from ray.train import Trainer
from ray.train.backend import BackendConfig, Backend
from ray.train.mindspore import MindSporeConfig

from ray.train.callbacks.callback import TrainingCallback
from ray.train._internal.worker_group import WorkerGroup


class TestConfig(BackendConfig):
    @property
    def backend_cls(self):
        return TestBackend

class TestBackend(Backend):
    def on_start(self, worker_group: WorkerGroup, backend_config: TestConfig):
        pass
    
    def on_shutdown(self, worker_group: WorkerGroup, backend_config: TestConfig):
        pass

class KillCallback(TrainingCallback):
    def __init__(self, fail_on, trainer):
        self.counter = 0
        self.fail_on = fail_on
        self.worker_group = trainer._backend_executor.get_worker_group()

    def handle_result(self, results):
        print(results)
        if self.counter == self.fail_on:
            ray.kill(self.worker_group.workers[0].actor)
            time.sleep(3)
        self.counter += 1

@pytest.mark.parameterize("backend", ["test", "mindspore"])
def test_worker_kill(ray_start_2_cpus, backend):
    if backend == "test":
        test_config = TestConfig()
    elif backend == "mindspore":
        test_config = MindSporeConfig()
    
    trainer = Trainer(test_config, num_workers=2)

    def train_func():
        for i in range(2):
            train.report(loss=1, iter=i)

    trainer.start()
    kill_callback = KillCallback(fail_on=0, trainer=trainer)
    trainer.run(train_func, callbacks=[kill_callback])
    # Run 1: iter=0, counter=1, Successful
    # Run 2: iter=1, counter=1, Unsuccessful, starts training from beginning
    # Run 3: iter=0, counter=2, Successful
    # Run 4: iter=1, counter=3, Successful
    assert kill_callback.counter == 3

    trainer.shutdown()
    trainer.start()

    kill_callback = KillCallback(fail_on=1, trainer=trainer)
    trainer.run(train_func, callbacks=[kill_callback])
    # Run 1: iter=0, counter=1, Successful
    # Run 2: iter=1, counter=2, Successful
    # Run 3: None, counter=2, Unsuccessful, starts training from beginning
    # Run 4: iter=0, counter=3, Successful
    # Run 5: iter=1, counter=4, Successful
    assert kill_callback.counter == 4

    def train_func():
        return 1

    # Make sure Trainer is usuable even after failure handling.
    trainer.run(train_func)

def test_mindspore_simple(ray_start_2_cpus):
    def simple_fn():
        #TODO
        pass

    num_workers = 2
    trainer = Trainer("mindspore", num_workers)
    trainer.start()
    result = trainer.run(simple_fn)
    trainer.shutdown()

    assert result == list(range(num_workers))