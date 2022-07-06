import setuptools

version = "0.1.0"

extras_require = {
    "latex": ("bibtexparser",)
}

setuptools.setup(
    name="ml-swissknife",
    packages=setuptools.find_packages(exclude=['experiments', 'templates', 'latex', 'tests']),
    version=version,
    license="MIT",
    description="Reusable ML research primitives for fast prototyping.",
    author="Xuechen Li",
    author_email="lxuechen@cs.toronto.edu",
    url="https://github.com/lxuechen/ml-swissknife",
    install_requires=[
        'torch', 'torchvision', 'spacy', 'tqdm', 'numpy', 'scipy', 'gputil', 'fire', 'requests', 'nltk', 'transformers',
        'datasets', 'gdown>=4.4.0', 'pandas', 'pytest', 'matplotlib', 'seaborn', 'cvxpy'
    ],
    extras_require=extras_require,
    python_requires='~=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
