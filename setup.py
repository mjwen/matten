from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NANE = "eigenn"


def get_version():
    version_dict = {}
    filename = Path(__file__).parent.joinpath(PACKAGE_NANE, "_version.py")
    with open(filename, "r") as f:
        exec(f.read(), version_dict)

    return version_dict["__version__"]


def get_readme():
    filename = Path(__file__).parent.joinpath("README.md")
    with open(filename, "r") as f:
        readme = f.read()
    return readme


setup(
    name=PACKAGE_NANE,
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "requests",
        "scipy",
        "pymatgen",
        "pytorch-lightning",
        "lightning-bolts",
        "loguru",
        "e3nn",
        "jsonargparse==3.19.2",
    ],
    extras_require={
        "test": ["pytest"],
    },
    author="Mingjian Wen",
    author_email="wenxx151@gmail.com",
    url="https://xxx.yyy.zzz",
    description="Short package description",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    zip_safe=False,
)
