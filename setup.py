import setuptools

version = "0.0.1"

setuptools.setup(
    name="swissknife",
    version=version,
    author="Xuechen Li",
    author_email="lxuechen@cs.toronto.edu",
    description="Reusable ML research primitives for fast prototyping. ",
    url="https://github.com/lxuechen/swissknife",
    packages=setuptools.find_packages(exclude=['experiments']),
    install_requires=['torch>=1.6.0'],
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
