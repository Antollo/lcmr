from setuptools import find_packages, setup

setup(
    name='lcmr',
    version='0.1.0',
    description='Learning Compositional Models via Reconstructions',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9.0',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'tensordict',
        'torchtyping',
        'typeguard<3.0.0',  # torchtyping requires typeguard<3.0.0
        'kornia',
        'moderngl',
        'pyefd',
        'scikit-learn',
        'torch_earcut @ git+https://github.com/Antollo/torch_earcut',
    ]
)