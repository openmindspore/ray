import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union

import numpy as np
import mindspore

from ray.train.mindspore.config import MindSporeConfig
from ray.train.trainer import GenDataset
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.air.config import ScalingConfig, RunConfig, DatasetConfig
from ray.air.checkpoint import Checkpoint
from ray.train._internal.dl_predictor import DLPredictor
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

@PublicAPI(stability="beta")
class MindSporeTrainer(DataParallelTrainer):
    """A Trainer for data parallel MindSpore training.
    
    This Trainer runs the function ``train_loop_per_worker`` on multiple Ray
    Actors. These actors already have the necessary mindspore proces gorup already
    configured for distributed MindSpore training.

    The ``train_loop_per_worker`` function is expected to take in either 0 or 1
    arguments:

    .. code-block:: python

        def train_loop_per_worker():
            ...
    .. code-block:: python

        def train_loop_per_worker(config: Dict):
            ...

    if ``train_loop_per_worker`` accepts an argument, then
    ``train_loop_config`` will be passed in as the argument. This is useful if you
    want to tune the values in ``train_loop_config`` as hyperparameters.

    If the ``datasets`` dict contains a training dataset (denoted by
    the "train" key), then it will be split into multiple dataset
    shards that can then be accesses by ``session.get_dataset_shard("train")`` inside
    ``train_loop_per_worker``. All the other datasets will not be split and
    ``session.get_dataset_shard( ... )`` will return the entire Dataset.

    Inside the ``train_loop_per_worker`` function, you can use any of the
    :ref:`Ray AIR session methods <air-session-ref>`.

    .. code-block:: python

        def train_loop_per_worker():
            # Report intermedia results for callbacks or logging and
            # checkpoint data.
            session.report( ... )

            # Returns dict of last saved checkpoint.
            session.get_checkpoint()

            # Returns the Ray Dataset shard for the given key.
            session.get_dataset_shard("my_dataset")

            # Returns the total number of workers executing training.
            session.get_world_size()

            # Returns the rank of this worker.
            session.get_world_rank()

            # Returns the rank of the worker on the current node.
            session.get_local_rank()

    You can also use any of the MindSpore specific function utils.

    """

    def __init__(
        self,
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        *,
        train_loop_config: Optional[Dict] = None,
        mindspore_config: Optional[MindSporeConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        dataset_config: Optional[Dict[str, DatasetConfig]] = None,
        run_config: Optional[Dict[str, GenDataset]] = None,
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