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
        'matplotlib'
    ]
)