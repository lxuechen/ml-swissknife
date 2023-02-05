import os
import re

import setuptools

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'ml_swissknife', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

extras_require = {
    "latex": ("bibtexparser",)
}

setuptools.setup(
    name="ml-swissknife",
    packages=setuptools.find_packages(exclude=['experiments', 'templates', 'latex', 'tests', 'turk']),
    version=version,
    license="MIT",
    description="Reusable ML research primitives for fast prototyping.",
    author="Xuechen Li",
    author_email="lxuechen@cs.toronto.edu",
    url="https://github.com/lxuechen/ml-swissknife",
    install_requires=[
        'torch', 'torchvision', 'spacy', 'tqdm', 'numpy', 'scipy', 'gputil', 'fire', 'requests', 'nltk', 'transformers',
        'datasets', 'gdown>=4.4.0', 'pandas', 'pytest', 'matplotlib', 'seaborn', 'cvxpy', 'imageio', 'wandb', 'openai',
        'numba'
    ],
    extras_require=extras_require,
    python_requires='~=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
