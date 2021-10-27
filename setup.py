import setuptools

version = "0.0.1"

setuptools.setup(
    name="swissknife",
    version=version,
    author="Xuechen Li",
    author_email="lxuechen@cs.toronto.edu",
    description="Reusable ML research primitives for fast prototyping. ",
    url="https://github.com/lxuechen/swissknife",
    packages=setuptools.find_packages(exclude=[]),
    install_requires=['torch', 'spacy', 'tqdm', 'numpy', 'scipy'],
    python_requires='~=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
