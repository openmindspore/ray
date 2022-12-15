import re
import pytest
import mindspore

import ray

from ray.air.checkpoint import Checkpoint
from ray.air.constants import MAX_REPR_LENGTH, MODEL_KEY, PREPROCESSOR_KEY
from ray.data.preprocessor import Preprocessor
from ray.train.mindspore import MindSporeCheckpoint, MindSporePredictor


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    # The code  after the yield will run as teardown code.
    ray.shutdown()

class DummyPreprocessor(Preprocessor):
    def transform_batch(self, df):
        return df * 2

class DummyModelSingleTensor(mindspore.nn.Cell):
    def forward(self, input):
        return input * 2

@pytest.fixture
def model():
    return DummyModelSingleTensor()

@pytest.fixture
def preprocessor():
    return DummyPreprocessor()

def test_repr(model):
    predictor = MindSporePredictor(model=model)

    representation = repr(predictor)

    assert len(representation) < MAX_REPR_LENGTH
    pattern = re.compile("^TorchPredictor\\((.*)\\)$")
    assert pattern.match(representation)

def test_init(model, preprocessor):
    predictor = MindSporePredictor(model=model, preprocessor=preprocessor)

    checkpoint = {MODEL_KEY: model, PREPROCESSOR_KEY: preprocessor}
    checkpoint_predictor = MindSporePredictor.from_checkpoint(
        Checkpoint.from_dict(checkpoint)
    )

    assert checkpoint_predictor.model == predictor.model
    assert checkpoint_predictor.get_preprocessor() == predictor.get_preprocessor()
