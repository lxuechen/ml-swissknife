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
    install_requires=[
        'torch', 'spacy', 'tqdm', 'numpy', 'scipy', 'gputil', 'fire', 'requests', 'nltk', 'transformers', 'datasets',
        'gdown>=4.4.0', 'pandas',
    ],
    python_requires='~=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
