from typing import Dict, List, Optional, Union

import mindspore

def load_mindspore_model(
    saved_model: Union[mindspore.nn.Cell, Dict],
    model_definition: Optional[mindspore.nn.Cell] = None,
) -> mindspore.nn.Cell:
    """Loads a MindSpore model from the provided ``saved_model``.
    
    If ``saved_model`` is a mindspore Cell, then return it directly. If ``saved_model`` is
    a mindspore state dict, then load it in the ``model_definition`` and return the loaded
    model.
    """
    #TODO
    pass
