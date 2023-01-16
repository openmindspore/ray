import ray
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

#ray.init()
ray.init(runtime_env={"working_dir": "https://www.dropbox.com/s/3ux02t02m2snuvl/torch.zip"})

class MyCell(nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
    def forward(self, x):
        x = F.relu(x)
        return x

def fn(x):
    model = MyCell()
    return model

remote_fn = ray.remote(fn)
res = remote_fn.remote(0)
print("res=", res)

futures = [remote_fn.remote(0) for i in range(2)]
print(ray.get(futures))

ray.shutdown()

