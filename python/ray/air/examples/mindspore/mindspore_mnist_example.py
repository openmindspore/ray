# This example showcases how to use MindSpore with Ray Train.
# Original code:
#
import argparse
from typing import Dict
from ray.air import session

from ray.train.mindspore import MindSporeTrainer
from ray.air.config import ScalingConfig

from mindspore import nn
from mindspore.common.initializer import TruncatedNormal

def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kenel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")
                     
def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

class NeuralNetwork(nn.Cell):
    def __init(self, num_class=10, channel=1):
        super(NeuralNetwork, self).__init()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
    
    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_func(config: Dict):
    batch_size = config["batch_isze"]
    lr = config["lr"]
    epochs = config["epochs"]
    
    # create model.
    model = NeuralNetwork(10)

    for i in range(epochs):
        #TODO
        print(i)

    
def train_mindspore_mnist(num_workers=1, use_gpu=False):
    trainer = MindSporeTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=True, help="Enables GPU training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    args, _ = parser.parse_known_args()

    import ray

    if args.smoke_test:
        # 1 workers + 1 for trainer.
        ray.init(num_cpus=2)
        train_mindspore_mnist()
    else:
        ray.init(address=args.address)
        train_mindspore_mnist(num_workers=args.num_workers, use_gpu=args.use_gpu)


