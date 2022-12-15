from typing import TYPE_CHECKING, Optional

import mindspore

from ray.air.checkpoint import Checkpoint
from ray.air.constants import MODEL_KEY, PREPROCESSOR_KEY
from ray.train.data_parallel_trainer import _load_checkpoint
from ray.air._internal.mindspore_utils import load_mindspore_model
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor

@PublicAPI(stability="beta")
class MindSporeCheckpoint(Checkpoint):
    """A: py:class:`~ray.air.checkpoint.Checkpoint` with MindSpore-specific
    functionaliry.

    create this from a generic :py:calss:`~ray.air.checkpoint.Checkpoint` by calling
    ``MindSporeCheckpoint.from__checkpoint(ckpt)``.
    """

    @classmethod
    def from_model(
        cls,
        model: mindspore.nn.Cell,
        *,
        preprocessor: Optional["Preprocessor"] = None,
    ) -> "MindSporeCheckpoint":
        """Create a :py:class:`~ray.air.checkpoint.Checkpoint` that stores a MindSpore
        model.
        
        Args:
            model: The MindSpore model to store in the checkpoint.
            preprocessor: A fitted preprocessor to be applied before inference.
        
        Returns:
            An :py:class:`MindSporeCheckpoint` containing the specified model.

        Examples:
            >>> from ray.train.mindspore import MindSporeCheckpoint
            >>> import mindspore
            >>>
            >>> model = ...
            >>> checkpoint = MindSporeCheckpoint.from_model(model)

            You can use a :py:class:`MidnSporeCheckpoint` to create an
            :py:class:`~ray.train.mindspore.MindSporePredictor` and preform inference.

            >>> from ray.train.mindspore import MindSporePredictor
            >>>
            >>> predictor = MindSporePredictor.from_checkpoint(checkpoint)            
        """
        checkpoint = cls.from_dict({PREPROCESSOR_KEY: preprocessor, MODEL_KEY: model})
        return checkpoint
    
    def get_model(self, model: Optional[mindspore.nn.Cell] = None) -> mindspore.nn.Cell:
        """Retrieve the model store in this checkpoint.
        
        Args:
            model: If the checkpoint contains a model state dict, and not
                the model itself, then the state dict will be loaded to this
                ``model``.
        """
        saved_model, _ = _load_checkpoint(self, "MindSporeTrainer")
        model = load_mindspore_model(saved_model=saved_model, model_definition=model)
        return model
