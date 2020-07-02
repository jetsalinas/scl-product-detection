from setuptools import find_packages
from setuptools import setup

setup(
    name='cloud-tpu',
    install_requires=["efficientnet", "gcsfs", "pandas", "sklearn", "numpy"],
    packages=find_packages()
)
