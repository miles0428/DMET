from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    user_options = []

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
    name="DMET-nolab",
    version="0.1.14",
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "numpy",
        "scipy",
        "openfermion",
        "cudaq",
        "tqdm"  # Added dependency
    ],
    entry_points={
        "console_scripts": [
            "dmet-hubbard=DMET.DMET:main"
        ]
    },
    author="Yu-Cheng Chung",
    description="A DMET implementation for the Hubbard model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/DMET-Hubbard",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    tests_require=["pytest"],
    cmdclass={"test": PyTest},
)
