# ml-swissknife

Reusable ML research primitives for fast prototyping.

Slowly porting and refactoring lots of stale code from many past projects.

## Installation

To install the stable release, run the command below.

```bash
pip install ml-swissknife
```

I occasionally may forget to push the latest release onto pypi, in which case running the command below will always
ensure that the latest repo is installed.

```bash
pip install git+https://github.com/lxuechen/ml-swissknife.git
```

## TODOs

- Numerical utils
    - [x] simultaneous iteration
    - [x] Lanczos
    - [x] spectral density plotting
    - [x] distributed simultaneous iteration
- NLP
    - [x] New token embedding for GPT-2
    - [x] Evaluation pipeline for generation
    - [x] Turk templates
- Misc
    - [ ] Fast two-sample test utils
    - [ ] Helper code for calibration metrics (reliability diagram, ECE, Brier score)
    - [ ] Custom optimizers (HF, K-FAC)
    - [x] Boilerplate for fine-tuning GPT-3 via OpenAI API.
    - [x] Confidence interval utils
- Data
    - [ ] UTKFaces
    - [ ] data2text
- wandb
    - [x] Project-based helper for downloading files
