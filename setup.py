from setuptools import find_packages, setup
from pathlib import Path

package_path = __file__
setup(
    name="reanalogy",
    version="0.0.1",
    author="Iordanis Fostiropoulos",
    author_email="mail@iordanis.me",
    url="https://iordanis.me",
    packages=find_packages(),
    install_requires=[
        "rstr",
    ],
    description="ReAnalogy",
    python_requires=">3.10",
    long_description=Path(package_path).parent.joinpath("README.md").read_text(),
)
