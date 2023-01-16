import ray
import numpy as np

#import mindspore
import mindspore.nn as nn

"""
ray.init(runtime_env={"py_modules": ["/home/wjeon/mindspore-ai/mindspore/output/mindspore_gpu-1.9.1-cp38-cp38-linux_x86_64.whl"],
                      "env_vars": {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
                     })
"""
"""
ray.init(runtime_env={"py_modules": ["/home/wjeon/mindspore-ai/mindspore/mindspore/python/mindspore",
                                     "/home/wjeon/mindspore-ai/mindspore/mindspore/python/mindspore/lib",
                                    ],
                      "excludes" : [#"/lib/libmindspore_backend.so",
                                    #"/lib/libopencv_core.so.4.5",
                                    #"/lib/libmindspore_grpc.so.15",
                                    #"/lib/libmindspore_grpc++.so.1",
                                    #"/lib/libmindspore_common.so",
                                    #"/lib/libmindspore_glog.so.0",
                                    "/lib/libmpi_adapter.so",
                                    #"/lib/libps_cache.so",
                                    #"/lib/libmindspore_gpr.so.15",
                                    "/lib/libnccl.so.2",
                                    "/lib/libakg.so",
                                    #"/lib/libdnnl.so.2",
                                    "/lib/libicui18n.so.69",
                                    "/lib/libnvidia_collective.so",
                                    #"/lib/libopencv_imgcodecs.so.4.5",
                                    #"/lib/libmindspore_core.so",
                                    "/lib/libicuuc.so.69",
                                    #"/lib/libmindspore_address_sorting.so.15",
                                    "/lib/libicudata.so.69",
                                    #"/lib/libmindspore_upb.so.15",
                                    "/lib/libgpu_collective.so",
                                    #"/lib/plugin/libmindspore_gpu.so",
                                    "/lib/libtinyxml2.so.8",
                                    "/lib/libcuda_ops.so",
                                    "/_mindspore_offline_debug.cpython-38-x86_64-linux-gnu.so",
                                    #"/_c_expression.cpython-38-x86_64-linux-gnu.so",
                                    #"/_c_dataengine.cpython-38-x86_64-linux-gnu.so",
                                    "/_c_mindrecord.cpython-38-x86_64-linux-gnu.so",
                                    "/_ms_mpi.cpython-38-x86_64-linux-gnu.so",
                                   ], 
})
"""
# ray.init(runtime_env={"working_dir": "https://10.145.87.82/mindspore.zip"})

ray.init(runtime_env={"working_dir": "https://www.dropbox.com/s/t1v6ltnbzpejzam/mindspore.zip"})

class MyCell(nn.Cell):
    def __init__(self):
        super(MyCell, self).__init__()
        self.relu = nn.ReLU()
    def construct(self, x):
        return self.relu(x)

def fn(x):
    model = MyCell()
    return model

remote_fn = ray.remote(fn)
# run remote fn once
res = remote_fn.remote(0)
print("res=", res)

# run remote fn 2 times
futures = [remote_fn.remote(0) for i in range(2)]
print(ray.get(futures))

ray.shutdown()
