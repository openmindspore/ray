"""
https://discuss.ray.io/t/runtimeerror-the-actor-with-name-traintrainable-failed-to-import-on-the-worker/8145
"""

from typing import TYPE_CHECKING, Callable, Dict, Optional, Union
from ray.air.checkpoint import Checkpoint
from ray.air.config import DatasetConfig, RunConfig, ScalingConfig
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train.trainer import GenDataset
from ray.util import PublicAPI

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

import ray
from ray.air.config import ScalingConfig

import io
from ray.train.backend import BackendConfig, Backend, EncodedData
from ray.train._internal.worker_group import WorkerGroup

import mindspore
from mindspore.nn import Cell

class MindSporeConfig(BackendConfig):
    backend: Optional[str] = None
    init_method: str = "env"
    timeout_s: int = 1800

    @property
    def backend_cls(self):
        return _MindSporeBackend


def _shutdown_mindspore(destroy_process_group=False):
    print("_shutdown_mindspore")


class _MindSporeBackend(Backend):
    share_cuda_visible_devices: bool = True

    def on_start(self, worker_group: WorkerGroup, backend_config: MindSporeConfig):
        print("_MindSporeBackend::on_start")

    def on_shutdown(elf, worker_group: WorkerGroup, backend_config: MindSporeConfig):
        worker_group.execute(
            _shutdown_mindspore, destroy_process_group=len(worker_group) > 1
        )

    @staticmethod
    def encode_data(data_dict: Dict) -> EncodedData:
        for k, v in data_dict.items():
            data_dict[k] = v.module

        _buffer = io.BytesIO()
        #mindspore.save_checkpoint(data_dict, _buffer)
        return _buffer.getvalue()

    @staticmethod
    def decode_data(encoded_data: EncodedData) -> Dict:
        _buffer = io.BytesIO(encoded_data)
        checkpoint_dict = mindspore.load(_buffer, map_location="cpu")
        return checkpoint_dict


class MindSporeTrainer(DataParallelTrainer):
    def __init__(
        self,
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        *,
        train_loop_config: Optional[Dict] = None,
        mindspore_config: Optional[MindSporeConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        dataset_config: Optional[Dict[str, DatasetConfig]] = None,
        run_config: Optional[RunConfig] = None,
        datasets: Optional[Dict[str, GenDataset]] = None,
        preprocessor: Optional["Preprocessor"] = None,
        resume_from_checkpoint: Optional[Checkpoint] = None,
    ):
        if not mindspore_config:
            mindspore_config = MindSporeConfig()

        super(MindSporeTrainer, self).__init__(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            backend_config=mindspore_config,
            scaling_config=scaling_config,
            dataset_config=dataset_config,
            run_config=run_config,
            datasets=datasets,
            preprocessor=preprocessor,
            resume_from_checkpoint=resume_from_checkpoint,
        )

import mindspore.nn as nn

class NeuralNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
    def construct(self, x):
        return self

scaling_config = ScalingConfig(num_workers=1)

def fn():
    torch_model = TorchNeuralNetwork()
    model = NeuralNetwork()

import torch.nn as torchnn
class TorchNeuralNetwork(torchnn.Module):
    def __init__(self):
        super(TorchNeuralNetwork, self).__init__()
        self.flatten = torchnn.Flatten()
        self.linear_relu_stack = torchnn.Sequential(
            torchnn.Linear(28 * 28, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 512),
            torchnn.ReLU(),
            torchnn.Linear(512, 10),
            torchnn.ReLU(),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        

# test pickling for nn.Cell
bin = ray.cloudpickle.dumps(fn)
ray.cloudpickle.loads(bin)

trainer = MindSporeTrainer(train_loop_per_worker=fn,
    scaling_config=scaling_config,
)
result = trainer.fit()
