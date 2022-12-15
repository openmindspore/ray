import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union

import numpy as np
import mindspore

from ray.air.checkpoint import Checkpoint
from ray.train.mindspore.mindspore_checkpoint import MindSporeCheckpoint
from ray.train._internal.dl_predictor import DLPredictor
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

@PublicAPI(stability="beta")
class MindSporePredictor(DLPredictor):
    """A predictor for MindSpore models.
    
    Args:
        model: The MindSpore module to use for predictions.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
        use_gpu: If set, the model will be moved to GPU on instantiation and
            prediction happens on GPU.
    """

    def __init__(
        self,
        model: mindspore.nn.Cell,
        preprocessor: Optional["Preprocessor"] = None,
        use_gpu: bool = False,
    ):
        self.model = model
        self.model.eval()

        self.use_gpu = use_gpu
        #TODO

        super().__init__(preprocessor)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model={self.model!r}, "
            f"preprocessor={self._preprocessor!r}, use_gpu={self.use_gpu!r})"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        model: Optional[mindspore.nn.Cell] = None,
        use_gpu: bool = False,
    ) -> "MindSporePredictor":
        """Instantiate the predictor from a Checkpoint.
        
        The checkpoint is expected to be a result of ``MindSporeTrainer``.

        Args:
            checkpoint: the checkpoint to load the model and
                preprocessor from. It is expected to be from the result of a
                ``MindSporeTrainer`` run.
            model: If the checkpoint contains a model state dict, and not
                the model itself, then the state dict will be loaded to this
                ``model``.
            use_gpu: If set, the model will be moved to GPU on instantiation and
                prediction happens on GPU.
        """
        checkpoint = MindSporeCheckpoint.from_checkpoint(checkpoint)
        model = checkpoint.get_model(model)

