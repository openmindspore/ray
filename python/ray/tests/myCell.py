import mindspore.nn as nn

class MyCell(nn.Cell):
    def __init__(self):
        super(MyCell, self).__init__()
        self.relu = nn.ReLU()
    def construct(self, x):
        return self.relu(x)
