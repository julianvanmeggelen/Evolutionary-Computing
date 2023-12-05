from setuptools import setup, find_packages

setup(
    name="hyper_parameter_optimization",
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
       'neat-python>=0.92'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)