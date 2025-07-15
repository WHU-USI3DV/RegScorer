from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='regscorer',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='regscorer.ext',
            sources=[
                'regscorer/extensions/extra/cloud/cloud.cpp',
                'regscorer/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'regscorer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'regscorer/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'regscorer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'regscorer/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
