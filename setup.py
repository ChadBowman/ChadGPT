from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = "0.1.0"

setup(
    name="chadgpt",
    version=VERSION,
    author="Chad Bowman",
    author_email="chad.bowman0+github@gmail.com",
    description="My own decoder-only transformer",
    long_description=README,
    long_description_type="text/markdown",
    url="https://github.com/ChadBowman/chadgpt",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "uvicorn ~= 0.21.1",
        "fastapi ~= 0.95.0",
        "python-multipart ~= 0.0.6",
        "torch >= 2.0.0",
        "numpy ~= 1.24.2",
    ]
)
